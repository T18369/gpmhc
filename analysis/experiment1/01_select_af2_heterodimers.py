#!/usr/bin/env python3
"""
Generate a training-only, diversity-maximized HLA-DQ peptide-MHCII structures
All peptides are restricted to the TRAIN split to prevent leakage

Strategy
1. Select HLA-DQ heterodimers from divergence analysis.
2. Select experimentally presented training peptides:
    - split == train
    - EL == 1
    - unique peptide cores
    - maximize peptide/interface diversity
3. Peptide diversity optimized using greedy farthest-point sampling across:
    - peptide sequence
    - peptide core sequence
    - peptide length
    - charge
    - hydrophobicity
    - aromaticity
    - polarity

Outputs
results/structure_selection/
    selected_dq_heterodimers.csv
    selected_dq_af2_complexes.csv
    dq_peptide_selection_summary.csv
    dq_peptide_selection_qc.csv
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

DIV_FILE = ("results/divergence/tables/divergence_vs_ap.csv")
PRESENTATION_FILE = ("Presentation_df_w_preds.csv")
OUT = Path("results/structure_selection")
OUT.mkdir(parents=True, exist_ok=True)

PEPTIDES_PER_HETERODIMER = 25

HYDROPHOBIC = set("AILMFWVY")
AROMATIC = set("FWY")
POLAR = set("NQST")
CHARGE = {
    "D": -1,
    "E": -1,
    "K": 1,
    "R": 1,
    "H": 0.5
}

def get_family(x):
    if str(x).startswith("DQA"):
        return "DQ"
    return "Other"

def extract_full_alleles(allotype):
    return [
        x.replace(" ","")
        for x in re.findall(
            r"(?:DQA1|DQB1)\s*\*\s*\d+:\d+",
            str(allotype)
        )
    ]

def canonical_dq(allotype):
    chains = extract_full_alleles(allotype)
    if len(chains) != 2:
        return None
    alpha = [x for x in chains if x.startswith("DQA1")]
    beta = [x for x in chains if x.startswith("DQB1")]
    if len(alpha) != 1 or len(beta) != 1:
        return None
    return (alpha[0] + "___" + beta[0])


def peptide_features(seq):
    seq = str(seq)
    length = len(seq)
    return {
        "length":length,
        "charge":sum(CHARGE.get(aa, 0)for aa in seq),
        "hydrophobic_fraction":sum(aa in HYDROPHOBIC for aa in seq) / length,
        "aromatic_fraction":sum(aa in AROMATIC for aa in seq) / length,
        "polar_fraction":sum(aa in POLAR for aa in seq) / length
    }

def sequence_distance(a,b):
    if len(a) != len(b):
        return abs(len(a)-len(b)) / max(len(a), len(b))
    return (sum(x != y for x,y in zip(a,b)) / len(a))

def feature_distance(a,b):
    cols = [
        "length",
        "charge",
        "hydrophobic_fraction",
        "aromatic_fraction",
        "polar_fraction"
    ]
    return np.mean(
        [
            abs(a[c]-b[c])
            for c in cols
        ]
    )

def peptide_distance(a,b):
    return (
        0.45 *
        sequence_distance(a["peptide"], b["peptide"])
        +
        0.30 *
        sequence_distance(a["peptide_core"], b["peptide_core"])
        +
        0.25 *
        feature_distance(a,b)
    )

print("\nLoading divergence metrics")
div = pd.read_csv(DIV_FILE)
div["family"] = (div["allotype"].apply(get_family))
div = div[div.family == "DQ"]
div = div.dropna(
    subset=[
        "AP",
        "nn_distance",
        "consensus_divergence"
    ]
)

print(f"DQ heterodimers available: {len(div)}")
selected = div.copy()
selected.to_csv(OUT / "selected_dq_heterodimers.csv",index=False)

print("\nLoading presentation dataset")
pred = pd.read_csv(PRESENTATION_FILE,low_memory=False)

pred = pred[pred["split"] == "train"]
print(f"Training presentation rows: {len(pred)}")

pred["heterodimer"] = (pred["allotype"].apply(canonical_dq))
selected_labels = set(selected["allotype"])
pred = pred[pred.heterodimer.isin(selected_labels)]
pred = pred[pred.EL == 1]
pred = pred.dropna(subset=["peptide", "peptide_core"])
print(f"Training DQ EL peptides: {len(pred)}")

selected_peptides = []
qc = []

for hla, group in pred.groupby("heterodimer"):
    group = (
        group
        .drop_duplicates("peptide_core")
        .copy()
    )

    features = pd.DataFrame(
        [
            peptide_features(p)
            for p in group.peptide
        ],
        index=group.index
    )

    group = pd.concat([group,features],axis=1)

    if len(group) <= PEPTIDES_PER_HETERODIMER:
        chosen = group
    else:
        # Random seed to avoid always selecting highest score peptide
        chosen = [group.sample(1,random_state=42).iloc[0]]
        remaining = group.drop(
            chosen[0].name
        )

        while len(chosen) < PEPTIDES_PER_HETERODIMER:
            scores = []
            for idx,row in remaining.iterrows():
                score = min(
                    [
                        peptide_distance(row, x)
                        for x in chosen
                    ]
                )
                scores.append(score)
            best = remaining.iloc[np.argmax(scores)]
            chosen.append(best)
            remaining = remaining.drop(best.name)
        chosen = pd.DataFrame(chosen)

    chosen["selection_rank"] = range(1,len(chosen)+1)

    selected_peptides.extend(chosen.to_dict("records"))
    qc.append(
        {
            "heterodimer": hla,
            "candidate_peptides":len(group),
            "selected_peptides":len(chosen)
        }
    )

af2_df = pd.DataFrame(selected_peptides)
if len(af2_df) == 0:
    raise RuntimeError("No DQ peptides selected")

af2_df = af2_df.merge(
    selected[
        [
            "allotype",
            "AP",
            "nn_distance",
            "consensus_divergence"
        ]
    ],
    left_on="heterodimer",
    right_on="allotype",
    how="left"
)

print("\nSplit verification:")
print(af2_df["split"].value_counts())
af2_df.to_csv(OUT / "selected_dq_af2_complexes.csv", index=False)

af2_df[
    [
        "heterodimer",
        "peptide",
        "peptide_core",
        "AP",
        "nn_distance"
    ]
].to_csv(OUT / "dq_peptide_selection_summary.csv", index=False)

pd.DataFrame(qc).to_csv(OUT / "dq_peptide_selection_qc.csv", index=False)

print("\nFinished")
print(f"AF2 complexes selected: {len(af2_df)}")
print(f"DQ heterodimers represented: " f"{af2_df.heterodimer.nunique()}")
print("\nPeptides per heterodimer:")
print(af2_df.groupby("heterodimer").size())
print("\nOutputs:")
print(OUT)
