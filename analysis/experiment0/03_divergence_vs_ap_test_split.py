#!/usr/bin/env python3

"""
03_divergence_vs_ap_test_split.py

Experiment 0.3: Pseudosequence Divergence vs Graph-pMHC Performance

Tests whether allotype-level Graph-pMHC performance is related to
pseudosequence similarity.

Two sequence-space measures are evaluated:
    1. Consensus divergence: normalized Hamming distance from the
       within-gene consensus pseudosequence.
    2. Nearest-neighbour distance: normalized Hamming distance to the
       most similar other allotype in the analysis set.

Statistics:
    - Spearman rank correlation (rho)
    - p-value
    - Linear regression slope

Input:
    results/tables/ap_by_allotype_test.csv
    gpmhc/mhc_seq_df.csv

Outputs:
    results/divergence/
        tables/
            divergence_vs_ap.csv
            divergence_correlation_summary.csv
        figures/
            divergence_vs_ap.png
            divergence_vs_ap_by_gene.png

Usage:
    python analysis/experiment2/03_divergence_vs_ap_test_split.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


AP_TABLE = Path("results/tables/ap_by_allotype_test.csv")
MHC_SEQ = Path("gpmhc/mhc_seq_df.csv")

OUT_ROOT = Path("results/divergence")
OUT_T = OUT_ROOT / "tables"
OUT_F = OUT_ROOT / "figures"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)


# Load the per-allotype Graph-pMHC TEST split results.
print("Loading Graph-pMHC TEST split AP table...")
ap_df = pd.read_csv(AP_TABLE)

print(f"Allotypes loaded: {len(ap_df)}")


# Load pseudosequences and identify the relevant columns without relying
# on exact column names in the source file.
print("\nLoading pseudosequences...")

mhc = pd.read_csv(MHC_SEQ, low_memory=False)

mhc.columns = (
    mhc.columns
    .str.strip()
    .str.lower()
)

allele_col = next(
    (c for c in mhc.columns if "allele" in c),
    None
)

seq_col = next(
    (
        c for c in mhc.columns
        if "pseudo" in c or "sequence" in c
    ),
    None
)

if allele_col is None or seq_col is None:
    raise ValueError(
        "Could not identify allele and pseudosequence columns "
        "in gpmhc/mhc_seq_df.csv"
    )

mhc = (
    mhc[[allele_col, seq_col]]
    .dropna()
    .copy()
)

mhc.columns = ["allele", "pseudo_seq"]

mhc["pseudo_seq"] = (
    mhc["pseudo_seq"]
    .astype(str)
    .str.strip()
)

seq_lookup = dict(
    zip(
        mhc["allele"],
        mhc["pseudo_seq"]
    )
)

print(f"Pseudosequences loaded: {len(seq_lookup)}")


def parse_chains(allotype):
    """Split a composite allotype into its component HLA chains."""
    return [
        x.strip()
        for x in str(allotype).split("___")
        if x.strip()
    ]


def concatenate_pseudo(chains):
    """Concatenate the pseudosequences for all chains in an allotype."""
    return "".join(
        seq_lookup.get(chain, "")
        for chain in chains
    )


ap_df["chains"] = ap_df["allotype"].apply(parse_chains)

ap_df["pseudo_concat"] = (
    ap_df["chains"]
    .apply(concatenate_pseudo)
)

missing = ap_df["pseudo_concat"].str.len() == 0

print(f"Missing pseudosequences: {missing.sum()}")

ap_df = ap_df[~missing].copy()


def gene_class(allotype):
    """Assign the allotype to the major MHC-II class."""
    x = str(allotype).upper()

    if "DRB" in x:
        return "DR"
    if "DPB" in x:
        return "DP"
    if "DQB" in x:
        return "DQ"

    return "OTHER"


ap_df["gene"] = ap_df["allotype"].apply(gene_class)


def hamming_norm(s1, s2):
    """Return normalized Hamming distance between two sequences."""
    length = min(len(s1), len(s2))

    if length == 0:
        return np.nan

    return sum(
        a != b
        for a, b in zip(s1[:length], s2[:length])
    ) / length


def consensus_sequence(seqs):
    """Build a position-wise consensus sequence."""
    max_len = max(len(s) for s in seqs)

    padded = [
        s.ljust(max_len, "-")
        for s in seqs
    ]

    consensus = []

    for i in range(max_len):
        column = [s[i] for s in padded]
        consensus.append(
            pd.Series(column).mode().iloc[0]
        )

    return "".join(consensus)


# Calculate divergence from the consensus separately within DP, DQ,
# and DR so the measure reflects sequence variation within each gene.
print("\nComputing consensus divergence...")

divergence_rows = []

for gene in ["DR", "DP", "DQ"]:

    sub = ap_df[ap_df["gene"] == gene].copy()

    if len(sub) < 3:
        continue

    consensus = consensus_sequence(
        sub["pseudo_concat"]
    )

    sub["consensus_divergence"] = (
        sub["pseudo_concat"]
        .apply(
            lambda x: hamming_norm(
                x,
                consensus
            )
        )
    )

    divergence_rows.append(sub)

    print(f"{gene}: n={len(sub)}")

if not divergence_rows:
    raise ValueError(
        "No gene classes contained enough allotypes "
        "to calculate divergence."
    )

ap_div = pd.concat(
    divergence_rows,
    ignore_index=True
)


# Calculate each allotype's distance to its closest neighbour in
# pseudosequence space.
print("\nComputing nearest-neighbour distances...")

seqs = ap_div["pseudo_concat"].tolist()
nn_distance = []

for i, seq in enumerate(seqs):

    distances = []

    for j, other in enumerate(seqs):

        if i == j:
            continue

        distance = hamming_norm(
            seq,
            other
        )

        if not np.isnan(distance):
            distances.append(distance)

    nn_distance.append(
        min(distances)
        if distances
        else np.nan
    )

ap_div["nn_distance"] = nn_distance


def calculate_spearman(dataframe, metric):
    """Calculate Spearman correlation between a sequence metric and AP."""
    valid = dataframe[
        [metric, "AP"]
    ].dropna()

    if len(valid) < 3:
        return np.nan, np.nan, len(valid)

    rho, p = spearmanr(
        valid[metric],
        valid["AP"]
    )

    return rho, p, len(valid)


def calculate_slope(dataframe, metric):
    """Calculate the ordinary least-squares slope of AP vs sequence metric."""
    valid = dataframe[
        [metric, "AP"]
    ].dropna()

    if len(valid) < 3:
        return np.nan

    slope, _ = np.polyfit(
        valid[metric],
        valid["AP"],
        1
    )

    return slope


# Calculate overall correlations and gene-specific consensus-divergence
# correlations.
print("\nSpearman correlations and regression slopes")

corr_rows = []

for metric in [
    "consensus_divergence",
    "nn_distance"
]:

    rho, p, n = calculate_spearman(
        ap_div,
        metric
    )

    slope = calculate_slope(
        ap_div,
        metric
    )

    print(
        f"{metric}: "
        f"rho={rho:.4f}, "
        f"p={p:.3e}, "
        f"slope={slope:.4f}, "
        f"n={n}"
    )

    corr_rows.append(
        {
            "metric": metric,
            "rho": rho,
            "p_value": p,
            "slope": slope,
            "n": n
        }
    )


for gene in ["DR", "DP", "DQ"]:

    sub = ap_div[
        ap_div["gene"] == gene
    ]

    rho, p, n = calculate_spearman(
        sub,
        "consensus_divergence"
    )

    slope = calculate_slope(
        sub,
        "consensus_divergence"
    )

    print(
        f"HLA-{gene}: "
        f"rho={rho:.4f}, "
        f"p={p:.3e}, "
        f"slope={slope:.4f}, "
        f"n={n}"
    )

    corr_rows.append(
        {
            "metric": f"consensus_divergence_{gene}",
            "rho": rho,
            "p_value": p,
            "slope": slope,
            "n": n
        }
    )


pd.DataFrame(corr_rows).to_csv(
    OUT_T / "divergence_correlation_summary.csv",
    index=False
)


# Save the allotype-level sequence divergence and AP values used for
# the analysis and figures.
ap_div[
    [
        "allotype",
        "gene",
        "n",
        "AP",
        "consensus_divergence",
        "nn_distance"
    ]
].sort_values("AP").to_csv(
    OUT_T / "divergence_vs_ap.csv",
    index=False
)


COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26",
    "OTHER": "#999999"
}


def add_regression_line(ax, dataframe, metric):
    """Add a simple linear regression line to an AP scatter plot."""
    valid = dataframe[
        [metric, "AP"]
    ].dropna()

    if len(valid) < 3:
        return

    slope, intercept = np.polyfit(
        valid[metric],
        valid["AP"],
        1
    )

    x_line = np.linspace(
        valid[metric].min(),
        valid[metric].max(),
        100
    )

    ax.plot(
        x_line,
        slope * x_line + intercept,
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.7
    )


def format_stats(dataframe, metric):
    """Format correlation statistics for use in plot titles."""
    rho, p, n = calculate_spearman(
        dataframe,
        metric
    )

    slope = calculate_slope(
        dataframe,
        metric
    )

    return (
        f"rho={rho:.3f}, "
        f"p={p:.2e}, "
        f"slope={slope:.3f}, "
        f"n={n}"
    )


# Plot overall AP against both measures of pseudosequence divergence.
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

for ax, metric, xlabel in zip(
    axes,
    [
        "consensus_divergence",
        "nn_distance"
    ],
    [
        "Consensus divergence",
        "Nearest-neighbour distance"
    ]
):

    for gene, color in COLORS.items():

        sub = ap_div[
            ap_div["gene"] == gene
        ]

        if len(sub) == 0:
            continue

        ax.scatter(
            sub[metric],
            sub["AP"],
            color=color,
            s=60,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"HLA-{gene}"
        )

    add_regression_line(
        ax,
        ap_div,
        metric
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        format_stats(
            ap_div,
            metric
        ),
        fontsize=10
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )


axes[0].legend(fontsize=8)

plt.suptitle(
    "Graph-pMHC TEST performance vs HLA-II pseudosequence divergence",
    fontsize=12
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    OUT_F / "divergence_vs_ap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Plot consensus divergence separately for each MHC-II gene class.
genes = [
    gene
    for gene in ["DR", "DP", "DQ"]
    if (ap_div["gene"] == gene).sum() >= 5
]

fig, axes = plt.subplots(
    1,
    len(genes),
    figsize=(5 * len(genes), 5),
    sharey=True
)

if len(genes) == 1:
    axes = [axes]

for ax, gene in zip(axes, genes):

    sub = ap_div[
        ap_div["gene"] == gene
    ]

    rho, p, n = calculate_spearman(
        sub,
        "consensus_divergence"
    )

    slope = calculate_slope(
        sub,
        "consensus_divergence"
    )

    ax.scatter(
        sub["consensus_divergence"],
        sub["AP"],
        color=COLORS[gene],
        s=65,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.4
    )

    add_regression_line(
        ax,
        sub,
        "consensus_divergence"
    )

    ax.set_title(
        f"HLA-{gene}\n"
        f"rho={rho:.3f}, "
        f"p={p:.2e}, "
        f"slope={slope:.3f}, "
        f"n={n}",
        fontsize=10
    )

    ax.set_xlabel("Consensus divergence")
    ax.set_ylim(0, 1.05)

    ax.grid(
        axis="y",
        alpha=0.3
    )

axes[0].set_ylabel("Average Precision (AP)")

plt.suptitle(
    "Per-gene pseudosequence divergence vs Graph-pMHC AP",
    fontsize=12
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)

plt.savefig(
    OUT_F / "divergence_vs_ap_by_gene.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nDone.")
print(f"Outputs saved to {OUT_ROOT}")
