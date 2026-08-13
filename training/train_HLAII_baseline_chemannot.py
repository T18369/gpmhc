#!/usr/bin/env python3
"""
Chemistry-annotation training script (PHASE 2 - parallel, non-destructive).
EXPERIMENT 003_HLAIIhet_chemannot.

This is a self-contained sibling of train_HLAII_baseline.py, not a
replacement for it. It imports the chemistry-specific architecture
(gpmhc.baseline_model_chemannot, which in turn imports
gpmhc.gnn_parts_chemannot) and defaults to json_input_chemannot.json.
Nothing in gnn_parts.py, baseline_model.py, json_input.json, or
train_HLAII_baseline.py is read, imported, or modified by this script -
the baseline pipeline that reproduces the released checkpoint and fine-
tunes to AP ~0.863 (experiment 002_HLAII_heterodimer) is completely
untouched and can still be run independently at any time.

Starting checkpoint for THIS experiment is 002's fine-tuned
HLAII_best.pth, NOT the released model_final.pth. This is deliberate: the
chemistry annotation is meant to be measured as a delta on top of the
heterodimer fine-tuning 002 already captured, not conflated with it. Using
model_final.pth here would let ordinary fine-tuning gains masquerade as
chemistry-annotation gains. The reference AP to compare against is
therefore ~0.863 (002's result), not ~0.82 (the released checkpoint).

Explicitly does NOT touch:
  - mhc_adj / graph topology / edge enumeration (chemistry features are
    appended to existing edges at forward-time in gnn_parts_chemannot.py,
    not baked into graph construction)

Known unresolved gaps (same as the baseline pipeline, inherited as-is):
  - loss_options.loss_func == "MaskedBCEWithLogitsLoss" is referenced by
    the config but not implemented anywhere in the provided codebase.
    This script uses plain BCEWithLogitsLoss instead.
  - mhc_seq_df.csv must be present at the working directory this script is
    run from (data.py's get_psuedos hardcodes a bare relative path).

CONFIRMED hard constraints (same as baseline pipeline):
  - GNN.forward() hardcodes .cuda() on node_feats and edge_feats regardless
    of module device - no CPU code path exists. This script requires CUDA.
  - Starting from 002's HLAII_best.pth checkpoint, exactly one layer is
    expected to show a shape mismatch and be reinitialized:
    init_context.project_edge1's input dimension changes from
    node_feat_size+3 to node_feat_size+9 because edge_feat_size changed.
    Everything else (including whatever heterodimer fine-tuning 002 did to
    the rest of the network) should load from the checkpoint with an exact
    match. Because that one layer starts randomly initialized, do not
    expect a --skip_training sanity check on this pipeline to reproduce
    ~0.863 AP before any training - that number is only meaningful for the
    untouched 002 checkpoint evaluated by its own (unmodified) architecture.
"""
import os
import sys
import json
import argparse
import importlib
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

from gpmhc.data import cleanup_schema, tokenize, dataset


def log(msg):
    print(msg, flush=True)


# =====================================================================
# Config
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Graph-pMHC HLA-II baseline training")
    p.add_argument("--config", type=str, default="models/baseline_model/json_input_chemannot.json")
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the starting checkpoint. For experiment "
                         "003_HLAIIhet_chemannot this should be 002_HLAII_heterodimer's "
                         "fine-tuned HLAII_best.pth (AP ~0.863), NOT the released "
                         "model_final.pth (AP ~0.82) - starting from the fine-tuned "
                         "checkpoint isolates the chemistry-annotation effect from "
                         "ordinary heterodimer fine-tuning gains 002 already captured.")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="checkpoints_chemannot")
    p.add_argument("--metric_dir", type=str, default="metrics_chemannot")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_training", action="store_true",
                    help="Only run the pre-training checkpoint sanity eval, then exit. "
                         "Use this first to confirm the loaded checkpoint reproduces "
                         "~0.863 AP before spending any epochs (against 002's fine-tuned checkpoint, not the released 0.82 baseline).")
    return p.parse_args()


def build_model(config):
    """Instantiate the model architecture directly, bypassing json_to_learner /
    fastai Learner entirely (confirmed unnecessary and non-functional as
    provided - Learner(data=None, ...) cannot drive training)."""
    arch_module = importlib.import_module(f"gpmhc.{config['arch']}")
    model_arch = arch_module.model(json_input=config)
    model = model_arch.get_model(config["model_hyper_opts"])
    return model, model_arch


def load_checkpoint(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path given but not found: {checkpoint_path}")

    log(f"Loading checkpoint: {checkpoint_path}")
    # map_location must match the device the model will actually run on -
    # GNN.forward hardcodes .cuda() internally regardless of module device,
    # so loading to "cpu" here and running forward later (as in the earlier
    # debug snippet) is exactly what produced the device-mismatch crash.
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Checkpoints seen so far in this codebase use either a raw state_dict,
    # a "model_state_dict" key (this script's own save format), or a "model"
    # key (seen in the debug snippet that hit the device-mismatch crash).
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    result_keys = set(state_dict.keys()) & set(model.state_dict().keys())
    shape_mismatches = [
        k for k in result_keys
        if state_dict[k].shape != model.state_dict()[k].shape
    ]
    if shape_mismatches:
        log(f"NOTE - {len(shape_mismatches)} key(s) have a shape mismatch and will be "
            f"reinitialized rather than loaded (expected if edge_feat_size changed, "
            f"e.g. project_edge1 in GetContext when adding chemistry features):")
        for k in shape_mismatches:
            log(f"    reinit (shape mismatch): {k}  "
                f"checkpoint={tuple(state_dict[k].shape)} model={tuple(model.state_dict()[k].shape)}")
        state_dict = {k: v for k, v in state_dict.items() if k not in shape_mismatches}

    result = model.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        log(f"WARNING - missing keys not found in checkpoint ({len(result.missing_keys)}):")
        for k in result.missing_keys:
            log(f"    missing: {k}")
    if result.unexpected_keys:
        log(f"WARNING - unexpected keys in checkpoint not used by model ({len(result.unexpected_keys)}):")
        for k in result.unexpected_keys:
            log(f"    unexpected: {k}")
    if not result.missing_keys and not result.unexpected_keys:
        log("Checkpoint loaded with an exact key match.")
    else:
        log("Checkpoint loaded with mismatches above - verify these are expected "
            "before trusting downstream metrics.")

    return model


# =====================================================================
# Evaluation
# =====================================================================
def evaluate(model, loader, device, tag="eval"):
    """Eval-mode GNN.forward returns (batch, 32): columns [0:16] are
    per-allele logits, columns [16:32] are per-allele graph indices
    (see baseline_model.py's non-training branch). Correct reduction is
    max over the 16 allele logits - NOT a fixed column index.

    tag distinguishes which call site this is in the printed log (e.g.
    the mandatory pre-training sanity check vs. a normal per-epoch eval) -
    this function previously had zero print statements despite running
    over the full test set BEFORE the first training batch, which is a
    likely place for a run to appear to hang silently.
    """
    print(f"[{tag}] START - {len(loader.dataset)} examples, "
          f"batch_size={loader.batch_size}", flush=True)
    _t0 = time.perf_counter()
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f"[{tag}] batch {batch_idx} starting "
                      f"(elapsed: {time.perf_counter() - _t0:.1f}s)", flush=True)
            xb = xb.to(device)

            print(f"[{tag}] batch {batch_idx}: before model(xb)", flush=True)
            _t_fwd_start = time.perf_counter()
            out = model(xb)
            print(f"[{tag}] batch {batch_idx}: after model(xb) "
                  f"({time.perf_counter() - _t_fwd_start:.3f}s)", flush=True)

            if out.ndim > 1:
                allele_logits = out[:, :16]
                logits = allele_logits.max(dim=1).values
            else:
                logits = out

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(yb.numpy())

    ap = average_precision_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    print(f"[{tag}] END total={time.perf_counter() - _t0:.1f}s  AP={ap:.4f}  AUC={auc:.4f}", flush=True)
    return ap, auc


# =====================================================================
# Main
# =====================================================================
def main():
    print("[main] SCRIPT STARTED", flush=True)
    args = parse_args()
    print(f"[main] args parsed: config={args.config} checkpoint={args.checkpoint} "
          f"epochs={args.epochs} batch_size={args.batch_size}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.metric_dir, exist_ok=True)

    print(f"[main] loading config from {args.config}", flush=True)
    with open(args.config) as f:
        config = json.load(f)
    print(f"[main] config loaded: arch={config.get('arch')} "
          f"edge_feat_size={config['model_hyper_opts'].get('edge_feat_size')} "
          f"use_chemistry_edges={config['model_hyper_opts'].get('use_chemistry_edges')}", flush=True)

    # baseline_model.py's GNN.forward() hardcodes .cuda() on node_feats/edge_feats
    # regardless of the module's actual device - there is no working CPU path in
    # the architecture as provided. Refuse to proceed rather than crash mid-forward.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. GNN.forward() hardcodes .cuda() internally "
            "and has no CPU code path - this will crash partway through the first "
            "forward pass. Switch the Colab runtime to a GPU runtime (Runtime > "
            "Change runtime type > GPU) before running this script."
        )
    device = torch.device("cuda")
    log(f"Device: {device}")

    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------------------
    # Model
    # -----------------------------
    # Order matters here: build on CPU, move to CUDA, THEN load the checkpoint
    # with map_location matching that same device. Loading with map_location
    #="cpu" and moving the model afterward (or not at all) is exactly what
    # produced the device-mismatch crash in the earlier debug run - the
    # hardcoded .cuda() calls inside GNN.forward assume the rest of the model
    # is already on CUDA too.
    print("[main] building model architecture (build_model)...", flush=True)
    _t_model_start = time.perf_counter()
    model, model_arch = build_model(config)
    _t_build_done = time.perf_counter()
    print(f"[main] build_model() done ({_t_build_done - _t_model_start:.1f}s) - "
          f"this includes nnalign_generate_combination_lookup_table(), which "
          f"precomputes every possible graph combinatorially and can itself "
          f"take real time on first construction", flush=True)

    model.to(device)
    print(f"[main] model.to(device) done ({time.perf_counter() - _t_build_done:.1f}s)", flush=True)

    print(f"[main] loading checkpoint from {args.checkpoint}", flush=True)
    _t_ckpt_start = time.perf_counter()
    model = load_checkpoint(model, args.checkpoint, device)
    print(f"[main] checkpoint loaded ({time.perf_counter() - _t_ckpt_start:.1f}s)", flush=True)

    # -----------------------------
    # Data
    # -----------------------------
    print(f"[main] reading train CSV: {args.train_csv}", flush=True)
    _t_csv_start = time.perf_counter()
    train_df = pd.read_csv(args.train_csv, low_memory=False)
    print(f"[main] train CSV read ({time.perf_counter() - _t_csv_start:.1f}s), shape={train_df.shape}", flush=True)

    print(f"[main] reading test CSV: {args.test_csv}", flush=True)
    _t_csv_start = time.perf_counter()
    test_df = pd.read_csv(args.test_csv, low_memory=False)
    print(f"[main] test CSV read ({time.perf_counter() - _t_csv_start:.1f}s), shape={test_df.shape}", flush=True)
    log(f"Train: {train_df.shape}  Test: {test_df.shape}")

    schema_options = config["dataloader_options"]["csv_to_df"]["schema_options"]

    # cleanup_schema -> get_psuedos reads mhc_seq_df.csv and does per-allotype
    # pandas lookups/string reconstruction. This has had ZERO print visibility
    # until now and is a real candidate for a silent multi-hour stall,
    # especially if mhc_seq_df.csv is large or if allotype matching is doing
    # more per-row work than expected. Watch this specific pair of prints.
    print("[main] cleanup_schema(train_df) starting - this reads mhc_seq_df.csv "
          "and does per-allotype sequence reconstruction; if execution stalls "
          "here for a long time, that function is the bottleneck, not anything "
          "in the model/training code.", flush=True)
    _t_schema_start = time.perf_counter()
    train_df = cleanup_schema(train_df, schema_options)
    print(f"[main] cleanup_schema(train_df) done ({time.perf_counter() - _t_schema_start:.1f}s)", flush=True)

    print("[main] cleanup_schema(test_df) starting...", flush=True)
    _t_schema_start = time.perf_counter()
    test_df = cleanup_schema(test_df, schema_options)
    print(f"[main] cleanup_schema(test_df) done ({time.perf_counter() - _t_schema_start:.1f}s)", flush=True)

    # NOTE: tokenize() only produces padded token-id tensors here. Graph
    # construction happens lazily inside GNN.forward via lookup_graph, using
    # model.arch.lookup_table (built once at model init from mhc_adj /
    # mhc_lens / bc_pad in json_input.json). No graph objects are created
    # at this stage - this is expected, not a gap.
    print("[main] tokenize(train_df) starting...", flush=True)
    _t_tok_start = time.perf_counter()
    x_train = tokenize(train_df, model_arch.tokenizer)
    print(f"[main] tokenize(train_df) done ({time.perf_counter() - _t_tok_start:.1f}s), shape={x_train.shape}", flush=True)
    y_train = train_df["EL"].values.astype("float32")

    print("[main] tokenize(test_df) starting...", flush=True)
    _t_tok_start = time.perf_counter()
    x_test = tokenize(test_df, model_arch.tokenizer)
    print(f"[main] tokenize(test_df) done ({time.perf_counter() - _t_tok_start:.1f}s), shape={x_test.shape}", flush=True)
    y_test = test_df["EL"].values.astype("float32")

    if x_train.min() < 0 or x_test.min() < 0:
        raise ValueError(
            "Negative token value found post-tokenization - likely an illegal "
            "character in a sequence column. Check cleanup_schema output before "
            "proceeding."
        )

    print("[main] constructing DataLoaders...", flush=True)
    train_loader = DataLoader(
        dataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    log("Dataloaders ready.")

    # -----------------------------
    # Pre-training sanity check (mandatory)
    # -----------------------------
    # Evaluate the loaded checkpoint BEFORE any optimizer step. This isolates
    # "does the checkpoint + reconstructed architecture reproduce ~0.863 AP on
    # its own" from "does further training help" - if this number is far from
    # 0.863, the problem is in checkpoint/architecture reconstruction, not in
    # anything the training loop does, and training further will not fix it.
    log("Running pre-training sanity check on loaded checkpoint...")
    print("[main] entering pretrain-sanity-eval over the FULL test set "
          f"({len(test_loader.dataset)} examples) - this runs BEFORE any "
          "training batch and was previously uninstrumented; watch the "
          "[pretrain-sanity-eval] prints below.", flush=True)
    pretrain_ap, pretrain_auc = evaluate(model, test_loader, device, tag="pretrain-sanity-eval")
    log(f"Checkpoint-only (no training) AP={pretrain_ap:.4f}  AUC={pretrain_auc:.4f}  "
        f"(reference: 002 fine-tuned checkpoint ~0.863)")
    if pretrain_ap < 0.5:
        log("WARNING: checkpoint-only AP is far below the ~0.863 reference. This points "
            "to a checkpoint/architecture mismatch (see missing/unexpected key warnings "
            "above) rather than a training-loop issue. Consider stopping here to debug "
            "before spending epochs on it.")

    if args.skip_training:
        log("skip_training set - exiting after sanity check.")
        return

    # -----------------------------
    # Optimizer / loss
    # -----------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Baseline-verification choice: plain BCEWithLogitsLoss, matching
    # baseline_model.py's own loss_fn (minus its max-over-alleles reduction,
    # which only applies to eval-shaped (batch, 32) output and is not
    # relevant here since the training branch returns (batch,) directly).
    # config['loss_options']['loss_func'] names "MaskedBCEWithLogitsLoss",
    # which is not implemented anywhere in the provided files - flagged
    # above, not silently substituted without notice.
    loss_fn = nn.BCEWithLogitsLoss()

    # -----------------------------
    # Training loop
    # -----------------------------
    metrics_log = []
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"[train] ===== EPOCH {epoch}/{args.epochs} START =====", flush=True)
        _epoch_t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (xb, yb) in enumerate(train_loader):
            _batch_t0 = time.perf_counter()
            if batch_idx % 10 == 0:
                print(f"[train] epoch {epoch} batch {batch_idx} starting "
                      f"(elapsed this epoch: {_batch_t0 - _epoch_t0:.1f}s)", flush=True)

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            print(f"[train] epoch {epoch} batch {batch_idx}: before model(xb)", flush=True)
            _t_fwd_start = time.perf_counter()
            logits = model(xb)
            _t_fwd_end = time.perf_counter()
            print(f"[train] epoch {epoch} batch {batch_idx}: after model(xb) "
                  f"({_t_fwd_end - _t_fwd_start:.3f}s)", flush=True)

            if logits.ndim > 1:
                logits = logits[:, :16].max(dim=1).values

            loss = loss_fn(logits, yb)
            print(f"[train] epoch {epoch} batch {batch_idx}: after loss computation "
                  f"loss={loss.item():.4f} ({time.perf_counter() - _t_fwd_end:.3f}s)", flush=True)

            _t_loss_computed = time.perf_counter()
            loss.backward()
            _t_backward_done = time.perf_counter()
            print(f"[train] epoch {epoch} batch {batch_idx}: after loss.backward() "
                  f"({_t_backward_done - _t_loss_computed:.3f}s)", flush=True)

            optimizer.step()
            _t_step_done = time.perf_counter()
            print(f"[train] epoch {epoch} batch {batch_idx}: after optimizer.step() "
                  f"({_t_step_done - _t_backward_done:.3f}s)  "
                  f"TOTAL BATCH TIME={_t_step_done - _batch_t0:.3f}s", flush=True)

            total_loss += loss.item()
            n_batches += 1

        print(f"[train] ===== EPOCH {epoch} END - {n_batches} batches in "
              f"{time.perf_counter() - _epoch_t0:.1f}s =====", flush=True)

        avg_loss = total_loss / max(n_batches, 1)
        ap, auc = evaluate(model, test_loader, device, tag=f"epoch{epoch}-eval")

        log(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  AP={ap:.4f}  AUC={auc:.4f}")

        metrics_log.append({"epoch": epoch, "loss": avg_loss, "AP": ap, "AUC": auc})

        ckpt_path = os.path.join(args.save_dir, f"HLAII_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "AP": ap,
                "AUC": auc,
                "config": config,
            },
            ckpt_path,
        )

        if ap > best_ap:
            best_ap = ap
            best_path = os.path.join(args.save_dir, "HLAII_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "AP": ap,
                    "AUC": auc,
                    "config": config,
                },
                best_path,
            )
            log(f"  New best AP ({ap:.4f}) - saved to {best_path}")

    metrics_df = pd.DataFrame(metrics_log)
    metrics_df.to_csv(os.path.join(args.metric_dir, "training_metrics.csv"), index=False)

    log(f"Training complete. Best test AP: {best_ap:.4f} (reference: 002 fine-tuned checkpoint ~0.863)")


if __name__ == "__main__":
    main()
