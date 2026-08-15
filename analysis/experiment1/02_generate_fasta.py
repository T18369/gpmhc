#!/usr/bin/env python3
"""
Generate AlphaFold2-Multimer FASTA inputs from selected DQ peptide-MHCII complexes.
Prepare AF2-Multimer FASTA files for structural training.

- Reads DQ peptide selection output
- Retrieves alpha/beta chain sequences
- Generates ColabFold multimer FASTA files
- Detects previously generated AF2 structures
- Allows incremental expansion of AF2 dataset

Previously generated structures are identified from:
    experiment1/af2_outputs/
using the naming convention: DQA1_01_01__DQB1_05_01__PEPTIDE

Outputs
experiment1/
    af2_inputs/
        all/
        DQ/
    af2_job_manifest.csv
"""

from pathlib import Path
import pandas as pd
from collections import defaultdict

SELECTED_COMPLEXES = Path(
    "results/structure_selection/"
    "selected_dq_af2_complexes_v3.csv"
)

CHAIN_FILE = Path("experiment1/sequences/allotype_chains.csv")
AF2_OUTPUTS = Path("experiment1/af2_outputs")
OUT_BASE = Path("experiment1/af2_inputs")
OUT_ALL = OUT_BASE / "all"
OUT_DQ = OUT_BASE / "DQ"

for d in [
    OUT_BASE,
    OUT_ALL,
    OUT_DQ
]:
    d.mkdir(parents=True, exist_ok=True)

MANIFEST_FILE = Path(
    "experiment1/"
    "af2_job_manifest.csv"
)

ALPHA_GENES = {"DQA1",}
BETA_GENES = {"DQB1",}

def parse_chains(heterodimer):
    return [
        x.strip()
        for x in str(
            heterodimer
        ).split("___")
        if x.strip()
    ]

def gene_root(chain):
    return chain.split("*")[0]

def classify_chains(chain_list):
    alpha = [
        c for c in chain_list
        if gene_root(c)
        in ALPHA_GENES
    ]
    beta = [
        c for c in chain_list
        if gene_root(c)
        in BETA_GENES
    ]

    if len(alpha) != 1 or len(beta) != 1:
        raise ValueError(f"Invalid DQ heterodimer: {chain_list}")
    return alpha[0], beta[0]

def clean_name(x):
    return (
        x
        .replace("*","_")
        .replace(":","_")
    )

def af2_prefix(heterodimer, peptide):
    return (
        clean_name(heterodimer)
        +
        "__"
        +
        peptide
    )

def existing_af2_structures():
    if not AF2_OUTPUTS.exists():
        return set()
    existing = set()
    for item in AF2_OUTPUTS.iterdir():
        if item.is_dir():
            existing.add(item.name)
    return existing

print("Loading DQ AF2 selections")
selected = pd.read_csv(SELECTED_COMPLEXES)
chains = pd.read_csv(CHAIN_FILE)

sequence_lookup = dict(
    zip(
        chains["chain"],
        chains["sequence"]
    )
)
print(f"Selected complexes: "f"{len(selected)}")
print(
    f"Heterodimers: "
    f"{selected.heterodimer.nunique()}"
)

existing = existing_af2_structures()

print(
    f"Existing AF2 outputs detected: "
    f"{len(existing)}"
)

jobs = []
skipped_existing = []
missing_sequences = []
counter = defaultdict(int)
print("\nGenerating AF2 inputs")

for _, row in selected.iterrows():
    heterodimer = row["heterodimer"]
    peptide = row["peptide"]
    peptide_core = row["peptide_core"]
    try:
        alpha_chain, beta_chain = classify_chains(
            parse_chains(heterodimer)
        )
    except ValueError as e:
        print(e)
        continue

    alpha_seq = sequence_lookup.get(alpha_chain)
    beta_seq = sequence_lookup.get(beta_chain)

    if alpha_seq is None or beta_seq is None:
        missing_sequences.append(
            {
                "heterodimer":heterodimer,
                "alpha":alpha_chain,
                "beta":beta_chain
            }
        )
        continue

    prefix = af2_prefix(heterodimer,peptide)

    # detect existing AF2 run
    already_exists = any(
        x.startswith(prefix)
        for x in existing
    )

    if already_exists:
        skipped_existing.append(prefix)
        continue
    counter[heterodimer] += 1

    job_name = prefix

    fasta = (
        f">{job_name}\n"
        f"{alpha_seq}:"
        f"{beta_seq}:"
        f"{peptide}\n"
    )

    out_file = (OUT_DQ / f"{job_name}.fasta")

    with open(
        out_file,
        "w"
    ) as f:
        f.write(fasta)

    all_file = (OUT_ALL / f"{job_name}.fasta")

    with open(
        all_file,
        "w"
    ) as f:
        f.write(fasta)

    jobs.append(
        {"job_name":job_name,
         "heterodimer":heterodimer,
         "alpha_chain":alpha_chain,
         "beta_chain":beta_chain,
         "peptide":peptide,
         "peptide_core":peptide_core,
         "AP":row["AP"],
         "nn_distance":row["nn_distance"],
         "alpha_length":len(alpha_seq),
         "beta_length":len(beta_seq),
         "fasta_path":str(all_file)
        }
    )

manifest = pd.DataFrame(jobs)
manifest.to_csv(MANIFEST_FILE,index=False)

print("AF2 INPUT SUMMARY")
print(f"\nNew FASTAs generated: " f"{len(manifest)}")
print(
    f"Skipped existing AF2 runs: "
    f"{len(skipped_existing)}"
)

print(
    f"Missing sequences: "
    f"{len(missing_sequences)}"
)

if len(manifest):
    print("\nNew structures per heterodimer:")
    print(
        manifest
        .groupby("heterodimer")
        .size()
    )

print("\nManifest:")
print(MANIFEST_FILE)
print("\nFinished.")
