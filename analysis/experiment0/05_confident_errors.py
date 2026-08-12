"""
05_confident_errors.py

Experiment 0.5 — High-Confidence Error Analysis
================================================

Identifies predictions where Graph-pMHC assigns an extreme score but
the ground truth label disagrees.

High-confidence false positive (hcFP):
    Score >= 90th percentile AND EL = 0

High-confidence false negative (hcFN):
    Score <= 10th percentile AND EL = 1

No classification threshold is applied. The analysis instead asks
whether the model makes confident mistakes at the extremes of its
score distribution.

Inputs:
    Presentation_df_w_preds.csv
    results_test_split/tables/ap_by_allotype_test.csv

Outputs:
    results/confident_errors/tables/high_confidence_errors.csv
    results/confident_errors/tables/top_confident_FP.csv
    results/confident_errors/tables/top_confident_FN.csv
    results/confident_errors/tables/confident_error_by_allotype.csv
    results/confident_errors/figures/confident_error_score_dist.png
    results/confident_errors/figures/confident_error_by_allotype.png
    results/confident_errors/figures/confident_error_by_gene.png

Usage:
    python analysis/05_confident_errors.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Paths
DATA = Path("Presentation_df_w_preds.csv")
AP_TABLE = Path("results_test_split/tables/ap_by_allotype_test.csv")
OUT_T = Path("results/confident_errors/tables")
OUT_F = Path("results/confident_errors/figures")

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

TOP_PERCENTILE = 90
BOT_PERCENTILE = 10


# Load prediction dataset
print("=" * 70)
print("Loading prediction dataset")
print("=" * 70)

df = pd.read_csv(
    DATA,
    usecols=[
        "EL",
        "Graph-pMHC_score",
        "allotype",
        "peptide"
    ],
    low_memory=False
)

print(
    f"Rows: {len(df):,} | "
    f"Allotypes: {df['allotype'].nunique()}"
)

df = df.dropna(
    subset=[
        "EL",
        "Graph-pMHC_score"
    ]
).copy()

print(
    f"Rows after dropping missing values: {len(df):,}"
)


# Classify HLA gene family
def parse_gene(allotype):

    x = str(allotype).upper()

    if "DRB" in x:
        return "DR"

    if "DPB" in x:
        return "DP"

    if "DQB" in x:
        return "DQ"

    return "OTHER"


df["gene"] = df["allotype"].apply(parse_gene)


# Define high-confidence score thresholds
p90 = df["Graph-pMHC_score"].quantile(
    TOP_PERCENTILE / 100
)

p10 = df["Graph-pMHC_score"].quantile(
    BOT_PERCENTILE / 100
)

print("\nScore distribution:")
print(
    f"Mean: {df['Graph-pMHC_score'].mean():.4f}"
)

print(
    f"Std: {df['Graph-pMHC_score'].std():.4f}"
)

print(
    f"10th percentile: {p10:.4f}"
)

print(
    f"90th percentile: {p90:.4f}"
)


# Identify high-confidence disagreements
hcFP_mask = (
    (df["Graph-pMHC_score"] >= p90) &
    (df["EL"] == 0)
)

hcFN_mask = (
    (df["Graph-pMHC_score"] <= p10) &
    (df["EL"] == 1)
)

hcFP = df[hcFP_mask].copy()
hcFN = df[hcFN_mask].copy()

hcFP["error_type"] = "hcFP"
hcFN["error_type"] = "hcFN"

print("\nHigh-confidence errors:")
print(
    f"hcFP: {len(hcFP):,}"
)

print(
    f"hcFN: {len(hcFN):,}"
)

print(
    f"Total: {len(hcFP) + len(hcFN):,}"
)


# Calculate error fractions
n_pos = (
    df["EL"] == 1
).sum()

n_neg = (
    df["EL"] == 0
).sum()

print(
    f"\nhcFP as % of all negatives: "
    f"{100 * len(hcFP) / n_neg:.2f}%"
)

print(
    f"hcFN as % of all positives: "
    f"{100 * len(hcFN) / n_pos:.2f}%"
)


# Save high-confidence error tables
all_errors = pd.concat(
    [
        hcFP,
        hcFN
    ],
    ignore_index=True
)

all_errors.to_csv(
    OUT_T / "high_confidence_errors.csv",
    index=False
)

hcFP.sort_values(
    "Graph-pMHC_score",
    ascending=False
).head(500).to_csv(
    OUT_T / "top_confident_FP.csv",
    index=False
)

hcFN.sort_values(
    "Graph-pMHC_score",
    ascending=True
).head(500).to_csv(
    OUT_T / "top_confident_FN.csv",
    index=False
)


# Join high-confidence errors with per-allotype AP
ap_df = pd.read_csv(
    AP_TABLE
)[
    [
        "allotype",
        "AP",
        "n",
        "gene"
    ]
]

fp_counts = (
    hcFP
    .groupby("allotype")
    .size()
    .reset_index(name="n_hcFP")
)

fn_counts = (
    hcFN
    .groupby("allotype")
    .size()
    .reset_index(name="n_hcFN")
)

allotype_counts = (
    df
    .groupby("allotype")
    .size()
    .reset_index(name="n_total")
)

err_by_allotype = (
    allotype_counts
    .merge(
        fp_counts,
        on="allotype",
        how="left"
    )
    .merge(
        fn_counts,
        on="allotype",
        how="left"
    )
    .merge(
        ap_df[
            [
                "allotype",
                "AP",
                "gene"
            ]
        ],
        on="allotype",
        how="left"
    )
    .fillna(
        {
            "n_hcFP": 0,
            "n_hcFN": 0
        }
    )
)

err_by_allotype["hcFP_rate"] = (
    err_by_allotype["n_hcFP"] /
    err_by_allotype["n_total"]
)

err_by_allotype["hcFN_rate"] = (
    err_by_allotype["n_hcFN"] /
    err_by_allotype["n_total"]
)

err_by_allotype["hc_error_rate"] = (
    (
        err_by_allotype["n_hcFP"] +
        err_by_allotype["n_hcFN"]
    ) /
    err_by_allotype["n_total"]
)

err_by_allotype.sort_values(
    "hc_error_rate",
    ascending=False
).to_csv(
    OUT_T / "confident_error_by_allotype.csv",
    index=False
)

print("\nTop 10 allotypes by high-confidence error rate")

print(
    err_by_allotype
    .sort_values(
        "hc_error_rate",
        ascending=False
    )
    .head(10)[
        [
            "allotype",
            "gene",
            "n_total",
            "n_hcFP",
            "n_hcFN",
            "hc_error_rate",
            "AP"
        ]
    ]
    .to_string(index=False)
)


# Summarize high-confidence errors by HLA gene
gene_summary = (
    err_by_allotype
    .groupby("gene")
    .agg(
        n_allotypes=("allotype", "count"),
        total_hcFP=("n_hcFP", "sum"),
        total_hcFN=("n_hcFN", "sum"),
        mean_AP=("AP", "mean"),
        mean_hc_error_rate=("hc_error_rate", "mean")
    )
    .reset_index()
)

print("\nGene-level high-confidence error summary")

print(
    gene_summary.to_string(
        index=False
    )
)


# Plotting configuration
GENE_COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26",
    "OTHER": "#aaaaaa"
}


# Figure 1: Score distribution and high-confidence error regions
fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.hist(
    df["Graph-pMHC_score"],
    bins=80,
    color="#cccccc",
    edgecolor="white",
    linewidth=0.3,
    label="All predictions",
    zorder=1
)

ax.hist(
    hcFP["Graph-pMHC_score"],
    bins=40,
    color="#d6604d",
    edgecolor="white",
    linewidth=0.3,
    alpha=0.8,
    label=f"hcFP (n={len(hcFP):,})",
    zorder=2
)

ax.hist(
    hcFN["Graph-pMHC_score"],
    bins=40,
    color="#2166ac",
    edgecolor="white",
    linewidth=0.3,
    alpha=0.8,
    label=f"hcFN (n={len(hcFN):,})",
    zorder=2
)

ax.axvline(
    p90,
    color="#d6604d",
    linestyle="--",
    linewidth=1.2,
    label=f"90th pct = {p90:.2f}"
)

ax.axvline(
    p10,
    color="#2166ac",
    linestyle="--",
    linewidth=1.2,
    label=f"10th pct = {p10:.2f}"
)

ax.set_xlabel(
    "Graph-pMHC logit score",
    fontsize=11
)

ax.set_ylabel(
    "Count",
    fontsize=11
)

ax.set_title(
    "High-confidence disagreements in Graph-pMHC predictions\n"
    "hcFP = high score + EL=0 | hcFN = low score + EL=1",
    fontsize=10
)

ax.legend(
    fontsize=9
)

ax.set_yscale(
    "log"
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confident_error_score_dist.png",
    dpi=300
)

plt.close()

print("\nSaved: confident_error_score_dist.png")


# Figure 2: High-confidence error rate versus AP
fig, ax = plt.subplots(
    figsize=(8, 5)
)

for gene, color in GENE_COLORS.items():

    sub = err_by_allotype[
        (err_by_allotype["gene"] == gene) &
        (err_by_allotype["AP"].notna())
    ]

    ax.scatter(
        sub["hc_error_rate"],
        sub["AP"],
        label=f"HLA-{gene} (n={len(sub)})",
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=55
    )

ax.set_xlabel(
    "High-confidence error rate (per allotype)",
    fontsize=11
)

ax.set_ylabel(
    "Average Precision (AP)",
    fontsize=11
)

ax.set_title(
    "High-confidence error rate vs AP per allotype\n"
    "(10th/90th score percentiles; no fixed classification threshold)",
    fontsize=10
)

ax.legend(
    fontsize=9
)

ax.set_ylim(
    0,
    1.05
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confident_error_by_allotype.png",
    dpi=300
)

plt.close()

print(
    "Saved: confident_error_by_allotype.png"
)


# Figure 3: High-confidence errors by HLA gene
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11, 4)
)

genes = gene_summary["gene"].tolist()

x = np.arange(
    len(genes)
)

for ax, column, label, color in [
    (
        axes[0],
        "total_hcFP",
        "hcFP count",
        "#d6604d"
    ),
    (
        axes[1],
        "total_hcFN",
        "hcFN count",
        "#2166ac"
    )
]:

    ax.bar(
        x,
        gene_summary[column],
        color=color,
        edgecolor="white",
        linewidth=0.5
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            f"HLA-{g}"
            for g in genes
        ],
        fontsize=10
    )

    ax.set_ylabel(
        "Count",
        fontsize=10
    )

    ax.set_title(
        label,
        fontsize=11
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

fig.suptitle(
    "High-confidence errors by HLA gene family",
    fontsize=10,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confident_error_by_gene.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: confident_error_by_gene.png"
)


# Final summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"\nTotal high-confidence disagreements: "
    f"{len(hcFP) + len(hcFN):,}"
)

print(
    f"hcFP: {len(hcFP):,} "
    f"({100 * len(hcFP) / n_neg:.1f}% of all negatives)"
)

print(
    f"hcFN: {len(hcFN):,} "
    f"({100 * len(hcFN) / n_pos:.1f}% of all positives)"
)

print(
    "\nInterpretation: High-confidence disagreements identify "
    "predictions at the extremes of the model score distribution "
    "where the ground-truth label disagrees. These errors are "
    "inconsistent with a simple borderline-uncertainty explanation "
    "and motivate examination of missing biological or structural "
    "information in the graph representation."
)

print(
    f"\nOutputs saved to {OUT_T.parent}"
)
