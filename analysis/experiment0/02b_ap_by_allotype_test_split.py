"""
02b_ap_by_allotype_test_split.py

Experiment 0.2b - Per-allotype performance atlas (test split)

Replicates the Graph-pMHC benchmark evaluation using the predefined
test split, then examines performance across individual allotypes
and MHC class.

AP is calculated from the raw Graph-pMHC logits without applying
a classification threshold.

Outputs:
    results/tables/
        overall_split_metrics.csv
        generalization_summary.txt
        ap_by_allotype_test.csv
        ap_by_molecule_test.csv
        low_performance_allotypes_test.csv
        high_performance_allotypes_test.csv
        ap_summary_test.txt

    results/figures/
        ap_by_allotype_test.png
        ap_distribution_test.png
        ap_by_molecule_test.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DATA = Path("Presentation_df_w_preds.csv") #source from zenodo ~1.4gb/1.4m predictions
OUT = Path("results")

OUT_T = OUT / "tables"
OUT_F = OUT / "figures"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 100

def parse_gene(allotype):
    """Assign an allotype to its HLA class II locus."""
    allotype = str(allotype).upper()

    if "DRB" in allotype:
        return "DR"
    if "DPB" in allotype:
        return "DP"
    if "DQB" in allotype:
        return "DQ"
    return "OTHER"

def molecule_class(allotype):
    """Assign multi-chain HLA annotations to DP, DQ, or DR."""
    genes = set()

    for allele in str(allotype).split("/"):
        if "*" in allele:
            genes.add(allele.split("*")[0])

    if any(g.startswith("DR") for g in genes):
        return "DR"
    if any(g.startswith("DQ") for g in genes):
        return "DQ"
    if any(g.startswith("DP") for g in genes):
        return "DP"

    return "OTHER"


print("=" * 60)
print("Loading Graph-pMHC prediction dataset")
print("=" * 60)

df = pd.read_csv(
    DATA,
    usecols=[
        "EL",
        "Graph-pMHC_score",
        "allotype",
        "split",
    ],
    low_memory=False,
)

print("\nDataset splits:")
print(df["split"].value_counts())


# Evaluate train/test performance before restricting to the test set.
# This provides a check of generalization before examining allotype-specific performance.

print("\n" + "=" * 60)
print("Overall Graph-pMHC generalization")
print("=" * 60)

overall_rows = []

for split_name, group in df.groupby("split"):
    if group["EL"].nunique() < 2:
        continue

    overall_rows.append(
        {
            "split": split_name,
            "n": len(group),
            "positives": int(group["EL"].sum()),
            "positive_fraction": group["EL"].mean(),
            "AP": average_precision_score(
                group["EL"],
                group["Graph-pMHC_score"],
            ),
            "ROC_AUC": roc_auc_score(
                group["EL"],
                group["Graph-pMHC_score"],
            ),
        }
    )

overall_df = (
    pd.DataFrame(overall_rows)
    .sort_values("split")
    .reset_index(drop=True)
)

print("\nOverall performance")
print("-" * 40)
print(overall_df.to_string(index=False))

overall_df.to_csv(
    OUT_T / "overall_split_metrics.csv",
    index=False,
)


if {"train", "test"}.issubset(set(overall_df["split"])):
    train = overall_df.loc[
        overall_df["split"] == "train"
    ].iloc[0]

    test = overall_df.loc[
        overall_df["split"] == "test"
    ].iloc[0]

    ap_gap = train["AP"] - test["AP"]
    auc_gap = train["ROC_AUC"] - test["ROC_AUC"]

    print("\nGeneralization")
    print("-" * 40)
    print(f"Train AP      : {train['AP']:.4f}")
    print(f"Test AP       : {test['AP']:.4f}")
    print(f"AP Gap        : {ap_gap:.4f}")
    print(f"Train ROC-AUC : {train['ROC_AUC']:.4f}")
    print(f"Test ROC-AUC  : {test['ROC_AUC']:.4f}")
    print(f"ROC-AUC Gap   : {auc_gap:.4f}")

    with open(OUT_T / "generalization_summary.txt", "w") as f:
        f.write("Graph-pMHC Generalization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training samples : {int(train['n']):,}\n")
        f.write(f"Test samples     : {int(test['n']):,}\n\n")
        f.write(f"Train AP         : {train['AP']:.5f}\n")
        f.write(f"Test AP          : {test['AP']:.5f}\n")
        f.write(f"AP Gap           : {ap_gap:.5f}\n\n")
        f.write(f"Train ROC-AUC    : {train['ROC_AUC']:.5f}\n")
        f.write(f"Test ROC-AUC     : {test['ROC_AUC']:.5f}\n")
        f.write(f"ROC-AUC Gap      : {auc_gap:.5f}\n")


# Allotype-level analysis uses the predefined test split.

df = df[df["split"] == "test"].copy()

print("\nTEST split")
print("-" * 40)
print(f"Rows: {len(df):,}")
print(f"Allotypes: {df['allotype'].nunique():,}")


# Per-allotype performance

print("\nCalculating per-allotype AP...")

rows = []

for allotype, group in df.groupby("allotype"):
    # Small groups and groups containing only one class are excluded
    # from per-allotype AP calculations.
    if len(group) < MIN_SAMPLES:
        continue

    if group["EL"].nunique() < 2:
        continue

    rows.append(
        {
            "allotype": allotype,
            "gene": parse_gene(allotype),
            "n": len(group),
            "n_pos": int(group["EL"].sum()),
            "n_neg": int((1 - group["EL"]).sum()),
            "positive_fraction": group["EL"].mean(),
            "AP": average_precision_score(
                group["EL"],
                group["Graph-pMHC_score"],
            ),
        }
    )

ap_df = (
    pd.DataFrame(rows)
    .sort_values("AP")
    .reset_index(drop=True)
)

ap_df.to_csv(
    OUT_T / "ap_by_allotype_test.csv",
    index=False,
)

print("\nAllotype performance")
print("-" * 40)
print(f"Evaluated allotypes: {len(ap_df)}")
print(f"Mean AP: {ap_df.AP.mean():.4f}")
print(f"Median AP: {ap_df.AP.median():.4f}")
print(f"Range: {ap_df.AP.min():.4f} - {ap_df.AP.max():.4f}")


# Define high- and low-performance groups using the lower and upper
# quintiles of the observed AP distribution.

low_cutoff = ap_df.AP.quantile(0.20)
high_cutoff = ap_df.AP.quantile(0.80)

ap_df[ap_df.AP <= low_cutoff].to_csv(
    OUT_T / "low_performance_allotypes_test.csv",
    index=False,
)

ap_df[ap_df.AP >= high_cutoff].to_csv(
    OUT_T / "high_performance_allotypes_test.csv",
    index=False,
)


# MHC class-level performance

print("\nCalculating molecule AP...")
df["molecule"] = df["allotype"].apply(molecule_class)
mol_rows = []
for molecule, group in df.groupby("molecule"):
    if len(group) < MIN_SAMPLES:
        continue

    if group["EL"].nunique() < 2:
        continue

    mol_rows.append(
        {
            "molecule": molecule,
            "n": len(group),
            "AP": average_precision_score(
                group["EL"],
                group["Graph-pMHC_score"],
            ),
        }
    )

mol_df = (
    pd.DataFrame(mol_rows)
    .sort_values("AP")
)

mol_df.to_csv(
    OUT_T / "ap_by_molecule_test.csv",
    index=False,
)


# Summary
with open(OUT_T / "ap_summary_test.txt", "w") as f:
    f.write("Graph-pMHC TEST SPLIT AP SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Rows: {len(df):,}\n")
    f.write(f"Allotypes: {df.allotype.nunique():,}\n")
    f.write(f"Evaluated allotypes: {len(ap_df)}\n\n")
    f.write(f"Mean AP: {ap_df.AP.mean():.5f}\n")
    f.write(f"Median AP: {ap_df.AP.median():.5f}\n")
    f.write(f"Std AP: {ap_df.AP.std():.5f}\n")
    f.write(f"Minimum AP: {ap_df.AP.min():.5f}\n")
    f.write(f"Maximum AP: {ap_df.AP.max():.5f}\n")


# Figures
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(
    ap_df.AP,
    bins=20,
    edgecolor="black",
)
ax.axvline(
    ap_df.AP.mean(),
    linestyle="--",
    color="gray",
)

ax.set_xlabel("Average Precision")
ax.set_ylabel("Number of allotypes")
ax.set_title(
    f"Graph-pMHC TEST split\n"
    f"AP distribution "
    f"(Mean={ap_df.AP.mean():.3f}; "
    f"Range={ap_df.AP.min():.3f}-{ap_df.AP.max():.3f})"
)

plt.tight_layout()

plt.savefig(
    OUT_F / "ap_distribution_test.png",
    dpi=300,
)

plt.close()


fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    range(len(ap_df)),
    ap_df.AP,
    marker="o",
    markersize=3,
    linewidth=1,
)

ax.axhline(
    ap_df.AP.mean(),
    linestyle="--",
    color="gray",
    label=f"Mean AP={ap_df.AP.mean():.3f}",
)

ax.set_xlabel("Allotypes ranked by AP")
ax.set_ylabel("Average Precision")
ax.set_ylim(0, 1.05)
ax.set_title("Graph-pMHC TEST split\nPer-allotype AP")
ax.legend()

plt.tight_layout()

plt.savefig(
    OUT_F / "ap_by_allotype_test.png",
    dpi=300,
)

plt.close()


fig, ax = plt.subplots(figsize=(6, 5))

ax.bar(
    mol_df.molecule,
    mol_df.AP,
    edgecolor="black",
)

ax.set_ylabel("Average Precision")
ax.set_ylim(0, 1.05)
ax.set_title("Graph-pMHC TEST split\nAP by molecule class")

plt.tight_layout()

plt.savefig(
    OUT_F / "ap_by_molecule_test.png",
    dpi=300,
)

plt.close()


print("\nLowest AP allotypes")
print(
    ap_df.head(10).to_string(index=False)
)

print("\nSaved:")
print(OUT)
