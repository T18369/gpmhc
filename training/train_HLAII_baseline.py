#!/usr/bin/env python3

"""
Baseline Graph-pMHC HLA-II training and checkpoint verification.

Phase 1 baseline only. This script does not modify:
    - chemistry-aware edge features
    - continuous distance features
    - mhc_adj / graph topology

The goal is to verify that the reconstructed baseline architecture and
pretrained checkpoint reproduce the expected Graph-pMHC performance before
any model modifications are introduced.

Known baseline limitations:
    - json_input.json references MaskedBCEWithLogitsLoss, but that loss is not
      implemented in the provided codebase. Plain BCEWithLogitsLoss is used.
    - gpmhc/data.py expects mhc_seq_df.csv at the working directory.
    - baseline_model.py hardcodes .cuda() inside GNN.forward(), so CUDA is
      required.
    - A previous random-initialization run reached only AP ~0.13-0.17 after
      20 epochs at lr=1e-5. The pretrained checkpoint should therefore be
      verified before training.

Usage:
    python analysis/experiment2/02_train_HLAII.py \
        --checkpoint path/to/model_final.pth \
        --train_csv path/to/train.csv \
        --test_csv path/to/test.csv

For checkpoint-only verification:
    python analysis/experiment2/02_train_HLAII.py \
        --checkpoint path/to/model_final.pth \
        --train_csv path/to/train.csv \
        --test_csv path/to/test.csv \
        --skip_training
"""

import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)

from gpmhc.data import cleanup_schema, dataset, tokenize


def log(message):
    print(message, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Graph-pMHC HLA-II baseline training and verification"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="models/baseline_model/json_input.json",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint.",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        required=True,
        help="Training dataset CSV.",
    )

    parser.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="Test dataset CSV.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory for model checkpoints.",
    )

    parser.add_argument(
        "--metric_dir",
        type=str,
        default="metrics",
        help="Directory for training metrics.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Evaluate the checkpoint only and exit.",
    )

    return parser.parse_args()


def build_model(config):
    """
    Build the Graph-pMHC architecture directly.

    The original fastai Learner construction is bypassed because the
    provided implementation does not supply a usable Learner/data setup
    for this training workflow.
    """
    arch_module = importlib.import_module(f"gpmhc.{config['arch']}")

    model_arch = arch_module.model(json_input=config)
    model = model_arch.get_model(config["model_hyper_opts"])

    return model, model_arch


def load_checkpoint(model, checkpoint_path, device):
    """Load a pretrained checkpoint into the reconstructed model."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    log(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    if result.missing_keys:
        log(
            f"WARNING: {len(result.missing_keys)} checkpoint keys "
            "are missing from the model."
        )
        for key in result.missing_keys:
            log(f"    missing: {key}")

    if result.unexpected_keys:
        log(
            f"WARNING: {len(result.unexpected_keys)} checkpoint keys "
            "were not used by the model."
        )
        for key in result.unexpected_keys:
            log(f"    unexpected: {key}")

    if not result.missing_keys and not result.unexpected_keys:
        log("Checkpoint loaded with an exact key match.")
    else:
        log(
            "Checkpoint loaded with key mismatches. "
            "Verify these before interpreting the results."
        )

    return model


def evaluate(model, loader, device):
    """
    Evaluate the model using Average Precision and ROC-AUC.

    In evaluation mode, GNN.forward() returns a tensor containing 16 allele
    logits followed by graph indices. The prediction for each observation
    is the maximum allele logit.
    """
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)

            output = model(x_batch)

            if output.ndim > 1:
                allele_logits = output[:, :16]
                logits = allele_logits.max(dim=1).values
            else:
                logits = output

            probabilities = torch.sigmoid(logits)

            predictions.extend(
                probabilities.cpu().numpy()
            )
            labels.extend(
                y_batch.numpy()
            )

    ap = average_precision_score(
        labels,
        predictions,
    )

    auc = roc_auc_score(
        labels,
        predictions,
    )

    return ap, auc


def prepare_data(args, config, model_arch):
    """Load, clean, tokenize, and batch the train/test datasets."""

    train_df = pd.read_csv(
        args.train_csv,
        low_memory=False,
    )

    test_df = pd.read_csv(
        args.test_csv,
        low_memory=False,
    )

    log(
        f"Train: {train_df.shape}  "
        f"Test: {test_df.shape}"
    )

    schema_options = (
        config["dataloader_options"]
        ["csv_to_df"]
        ["schema_options"]
    )

    train_df = cleanup_schema(
        train_df,
        schema_options,
    )

    test_df = cleanup_schema(
        test_df,
        schema_options,
    )

    x_train = tokenize(
        train_df,
        model_arch.tokenizer,
    )

    x_test = tokenize(
        test_df,
        model_arch.tokenizer,
    )

    y_train = train_df["EL"].values.astype("float32")
    y_test = test_df["EL"].values.astype("float32")

    if x_train.min() < 0 or x_test.min() < 0:
        raise ValueError(
            "Negative token values detected after tokenization. "
            "Check the cleaned sequence columns."
        )

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

    return train_loader, test_loader


def train(
    model,
    train_loader,
    test_loader,
    device,
    config,
    args,
):
    """Run baseline training and save epoch checkpoints."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # json_input.json references MaskedBCEWithLogitsLoss, but that loss is
    # not implemented in the supplied codebase. Use the baseline BCE loss
    # explicitly rather than silently assuming another implementation.
    loss_fn = nn.BCEWithLogitsLoss()

    metrics = []
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(x_batch)

            if output.ndim > 1:
                logits = output[:, :16].max(dim=1).values
            else:
                logits = output

            loss = loss_fn(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)

        ap, auc = evaluate(
            model,
            test_loader,
            device,
        )

        log(
            f"Epoch {epoch}/{args.epochs}  "
            f"loss={mean_loss:.4f}  "
            f"AP={ap:.4f}  "
            f"AUC={auc:.4f}"
        )

        metrics.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "AP": ap,
                "AUC": auc,
            }
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "AP": ap,
            "AUC": auc,
            "config": config,
        }

        epoch_path = os.path.join(
            args.save_dir,
            f"HLAII_epoch_{epoch}.pth",
        )

        torch.save(
            checkpoint,
            epoch_path,
        )

        if ap > best_ap:
            best_ap = ap

            best_path = os.path.join(
                args.save_dir,
                "HLAII_best.pth",
            )

            torch.save(
                checkpoint,
                best_path,
            )

            log(
                f"  New best AP: {ap:.4f} "
                f"-> {best_path}"
            )

    metrics_df = pd.DataFrame(metrics)

    metrics_df.to_csv(
        os.path.join(
            args.metric_dir,
            "training_metrics.csv",
        ),
        index=False,
    )

    log(
        f"Training complete. "
        f"Best test AP: {best_ap:.4f}"
    )


def main():
    args = parse_args()

    os.makedirs(
        args.save_dir,
        exist_ok=True,
    )

    os.makedirs(
        args.metric_dir,
        exist_ok=True,
    )

    with open(args.config) as handle:
        config = json.load(handle)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. "
            "baseline_model.py uses .cuda() inside GNN.forward(), "
            "so this implementation requires a GPU runtime."
        )

    device = torch.device("cuda")
    log(f"Device: {device}")

    seed = config.get("seed", 0)

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Build the architecture and move it to CUDA before loading the
    # checkpoint. GNN.forward() assumes CUDA tensors internally.
    model, model_arch = build_model(config)
    model.to(device)

    load_checkpoint(
        model,
        args.checkpoint,
        device,
    )

    train_loader, test_loader = prepare_data(
        args,
        config,
        model_arch,
    )

    # Verify the supplied checkpoint before modifying its weights.
    log(
        "Evaluating pretrained checkpoint "
        "before training..."
    )

    pretrain_ap, pretrain_auc = evaluate(
        model,
        test_loader,
        device,
    )

    log(
        f"Checkpoint AP={pretrain_ap:.4f}  "
        f"AUC={pretrain_auc:.4f}  "
        f"(reference baseline ~0.82 AP)"
    )

    if pretrain_ap < 0.5:
        log(
            "WARNING: checkpoint AP is substantially below "
            "the expected baseline. Check the architecture, "
            "checkpoint keys, configuration, and preprocessing "
            "before proceeding with training."
        )

    if args.skip_training:
        log(
            "Checkpoint-only evaluation complete. "
            "Training skipped."
        )
        return

    train(
        model,
        train_loader,
        test_loader,
        device,
        config,
        args,
    )


if __name__ == "__main__":
    main()
