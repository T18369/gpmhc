#!/usr/bin/env python3

"""
Compare NetMHCIIpan performance across MHC-II allotypes.

This uses the same TEST split and per-allotype AP calculation
as the Graph-pMHC analysis so the two models can be compared
directly.

NetMHCIIpan reports percentile rank rather than a prediction
score. Lower rank indicates stronger predicted binding, so the
rank is multiplied by -1 before calculating AP. This only reverses
the ordering and therefore preserves the information used by
average precision.

Input:
    Presentation_df_w_preds.csv

Required columns:
    EL
    NetMHCIIPan-4.0
    allotype
    split

Outputs:
    results_netmhcpan_test_split/
        tables/
            ap_by_allotype_netmhcpan_test.csv
            ap_by_molecule_netmhcpan_test.csv
            ap_summary_netmhcpan_test.txt

        figures/
            ap_by_allotype_netmhcpan_test.png
            ap_distribution_netmhcpan_test.png

Run from the repository root:
    python analysis/experiment2/02c_netmhcpan_ap_by_allotype_test_split.py
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score


DATA = Path("Presentation_df_w_preds.csv")

OUT = Path("results_netmhcpan_test_split")
OUT_T = OUT / "tables"
OUT_F = OUT / "figures"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 100


# Load the same prediction dataset used for the Graph-pMHC analysis.
df = pd.read_csv(
    DATA,
    usecols=[
        "EL",
        "NetMHCIIPan-4.0",
        "allotype",
        "split"
    ],
    low_memory=False
)

print("=" * 70)
print("Loading NetMHCIIpan prediction dataset")
print("=" * 70)

print(f"\nRows: {len(df):,}")
print(f"Allotypes: {df['allotype'].nunique():,}")

print("\nDataset splits:")
print(df["split"].value_counts())


# Use the predefined TEST split so performance can be compared
# directly with the Graph-pMHC test-set analysis.
df = df[
    df["split"].astype(str).str.lower() == "test"
].copy()

df = df.dropna(
    subset=[
        "EL",
        "NetMHCIIPan-4.0"
    ]
)

print("\nTEST split")
print("-" * 40)
print(f"Rows: {len(df):,}")
print(f"Allotypes: {df['allotype'].nunique():,}")


# NetMHCIIpan reports percentile rank:
# lower rank = stronger predicted binding.
#
# Reverse the sign so that higher values correspond to stronger
# predictions, matching the orientation of Graph-pMHC scores.
df["NetMHCIIPan_score"] = -df["NetMHCIIPan-4.0"]


def parse_gene(allotype):

    allotype = str(allotype).upper()

    if "DRB" in allotype:
        return "DR"

    if "DPB" in allotype:
        return "DP"

    if "DQB" in allotype:
        return "DQ"

    return "OTHER"


def molecule_class(allotype):

    genes = sorted(
        set(
            token.split("*")[0]
            for token in str(allotype).split("___")
            if "*" in token
        )
    )

    return ",".join(genes)


# Per-allotype AP
#
# The minimum sample threshold matches the Graph-pMHC analysis.
# Allotypes without both positive and negative examples cannot
# provide a meaningful AP estimate and are skipped.
print("\nCalculating per-allotype AP...")

rows = []

for allotype, group in df.groupby("allotype"):

    if len(group) < MIN_SAMPLES:
        continue

    if group["EL"].nunique() < 2:
        continue

    rows.append({
        "allotype": allotype,
        "gene": parse_gene(allotype),
        "n": len(group),
        "n_pos": int(group["EL"].sum()),
        "n_neg": int((1 - group["EL"]).sum()),
        "positive_fraction": group["EL"].mean(),
        "AP": average_precision_score(
            group["EL"],
            group["NetMHCIIPan_score"]
        )
    })


ap_df = (
    pd.DataFrame(rows)
    .sort_values("AP")
    .reset_index(drop=True)
)


print("\nAllotype performance")
print("-" * 40)
print(f"Valid allotypes: {len(ap_df)}")
print(f"Mean AP: {ap_df['AP'].mean():.4f}")
print(f"Median AP: {ap_df['AP'].median():.4f}")
print(f"Std AP: {ap_df['AP'].std():.4f}")
print(
    f"Range: "
    f"{ap_df['AP'].min():.4f} - "
    f"{ap_df['AP'].max():.4f}"
)


ap_df.to_csv(
    OUT_T / "ap_by_allotype_netmhcpan_test.csv",
    index=False
)


# Molecule-level AP
df["molecule"] = df["allotype"].apply(
    molecule_class
)

print("\nCalculating molecule AP...")

mol_rows = []

for molecule, group in df.groupby("molecule"):

    if len(group) < MIN_SAMPLES:
        continue

    if group["EL"].nunique() < 2:
        continue

    mol_rows.append({
        "molecule": molecule,
        "n": len(group),
        "AP": average_precision_score(
            group["EL"],
            group["NetMHCIIPan_score"]
        )
    })


mol_df = (
    pd.DataFrame(mol_rows)
    .sort_values("AP")
)


mol_df.to_csv(
    OUT_T / "ap_by_molecule_netmhcpan_test.csv",
    index=False
)


# Save a small text summary alongside the tables.
with open(
    OUT_T / "ap_summary_netmhcpan_test.txt",
    "w"
) as f:

    f.write("NetMHCIIpan TEST SPLIT AP SUMMARY\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Rows: {len(df):,}\n")
    f.write(f"Valid allotypes: {len(ap_df)}\n\n")

    f.write(f"Mean AP: {ap_df['AP'].mean():.5f}\n")
    f.write(f"Median AP: {ap_df['AP'].median():.5f}\n")
    f.write(f"Std AP: {ap_df['AP'].std():.5f}\n")
    f.write(f"Minimum AP: {ap_df['AP'].min():.5f}\n")
    f.write(f"Maximum AP: {ap_df['AP'].max():.5f}\n")


# Per-allotype figure.
#
# Show the lowest-performing allotypes and highest-performing
# allotypes rather than trying to label the entire set.
display_df = pd.concat([
    ap_df.head(20),
    ap_df.tail(10)
]).drop_duplicates("allotype")


fig, ax = plt.subplots(
    figsize=(14, 8)
)

ax.barh(
    range(len(display_df)),
    display_df["AP"],
    edgecolor="white"
)

ax.set_yticks(
    range(len(display_df))
)

ax.set_yticklabels(
    display_df["allotype"],
    fontsize=8
)

ax.axvline(
    ap_df["AP"].mean(),
    linestyle="--",
    color="gray"
)

ax.set_xlim(
    0,
    1.05
)

ax.set_xlabel(
    "Average Precision"
)

ax.set_title(
    "NetMHCIIpan TEST split\nPer-allotype AP"
)

plt.tight_layout()

plt.savefig(
    OUT_F / "ap_by_allotype_netmhcpan_test.png",
    dpi=300
)

plt.close()


# AP distribution.
fig, ax = plt.subplots(
    figsize=(7, 4)
)

ax.hist(
    ap_df["AP"],
    bins=20,
    edgecolor="white"
)

ax.axvline(
    ap_df["AP"].mean(),
    linestyle="--",
    color="gray"
)

ax.set_xlabel(
    "Average Precision"
)

ax.set_ylabel(
    "Number of allotypes"
)

ax.set_title(
    "NetMHCIIpan TEST split\nAP distribution"
)

plt.tight_layout()

plt.savefig(
    OUT_F / "ap_distribution_netmhcpan_test.png",
    dpi=300
)

plt.close()


print("\nLowest AP allotypes")
print(
    ap_df[
        [
            "allotype",
            "gene",
            "n",
            "n_pos",
            "AP"
        ]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nSaved:")
print(OUT)
