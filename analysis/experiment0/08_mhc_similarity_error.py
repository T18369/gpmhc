"""
08_mhc_similarity_error.py
Experiment 0.8 — MHC Pseudosequence Similarity vs Performance

Tests whether high-error allotypes occupy a different region of MHC
pseudosequence space than low-error allotypes.

High-error allotypes: Bottom 20% by test-set AP.
Low-error allotypes: Top 20% by test-set AP.

The analysis asks whether poor performance is associated with:
    1. High similarity among poorly performing allotypes
    2. Low similarity / sequence-space isolation

Inputs:
    results_test_split/tables/ap_by_allotype_test.csv
    gpmhc/mhc_seq_df.csv

Outputs:
    results/mhc_similarity/tables/mhc_similarity_error.csv
    results/mhc_similarity/figures/mhc_similarity_boxplot.png
    results/mhc_similarity/figures/mhc_similarity_vs_ap.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu, spearmanr

AP_TABLE = Path("results_test_split/tables/ap_by_allotype_test.csv")
MHC_FILE = Path("gpmhc/mhc_seq_df.csv")
OUT_T = Path("results/mhc_similarity/tables")
OUT_F = Path("results/mhc_similarity/figures")
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

HIGH_ERROR_CUTOFF = 0.20
LOW_ERROR_CUTOFF = 0.80

print("Loading test-set AP table")
ap_df = pd.read_csv(AP_TABLE)

print(f"Allotypes: {len(ap_df)}")
print(
    f"AP range: "
    f"{ap_df['AP'].min():.4f} – "
    f"{ap_df['AP'].max():.4f}"
)

# Define high- and low-performance groups
ap_thresh_low = ap_df["AP"].quantile(HIGH_ERROR_CUTOFF)
ap_thresh_high = ap_df["AP"].quantile(LOW_ERROR_CUTOFF)
high_error = ap_df[ap_df["AP"] <= ap_thresh_low].copy()
low_error = ap_df[ap_df["AP"] >= ap_thresh_high].copy()

print(
    f"\nHigh-error group "
    f"(AP ≤ {ap_thresh_low:.4f}): "
    f"{len(high_error)} allotypes"
)
print(
    f"Low-error group "
    f"(AP ≥ {ap_thresh_high:.4f}): "
    f"{len(low_error)} allotypes"
)
print("\nHigh-error allotypes:")

for _, row in high_error.sort_values("AP").iterrows():
    print(
        f"  {row['allotype']:<55} "
        f"AP={row['AP']:.4f}"
    )

print(f"\nLoading MHC pseudosequences from {MHC_FILE}")
mhc = pd.read_csv(MHC_FILE, low_memory=False)
mhc.columns = (
    mhc.columns
    .str.strip()
    .str.lower()
)

allele_col = next(
    (
        c for c in mhc.columns
        if "allele" in c
    ),
    None
)

seq_col = next(
    (
        c for c in mhc.columns
        if "pseudo" in c
        or "sequence" in c
    ),
    None
)

if allele_col is None or seq_col is None:

    raise ValueError(
        "Could not identify allele and pseudosequence "
        f"columns. Available columns: {mhc.columns.tolist()}"
    )

mhc = mhc[
    [
        allele_col,
        seq_col
    ]
].dropna()

mhc = mhc.rename(
    columns={
        allele_col: "allele",
        seq_col: "pseudo_seq"
    }
)

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


# Build concatenated pseudosequence for each allotype
def parse_chains(allotype):
    return [
        x.strip()
        for x in str(allotype).split("___")
        if x.strip()
    ]


def concat_pseudo(allotype):

    chains = parse_chains(
        allotype
    )

    return "".join(
        seq_lookup.get(
            chain,
            ""
        )
        for chain in chains
    )


ap_df["pseudo_concat"] = (
    ap_df["allotype"]
    .apply(concat_pseudo)
)

high_error = ap_df[
    ap_df["AP"] <= ap_thresh_low
].copy()

low_error = ap_df[
    ap_df["AP"] >= ap_thresh_high
].copy()


# Check for missing pseudosequences
for group_name, group in [
    ("high-error", high_error),
    ("low-error", low_error)
]:

    missing = (
        group["pseudo_concat"]
        .str.len()
        == 0
    )

    if missing.sum():

        print(
            f"Warning: {missing.sum()} "
            f"{group_name} allotypes have no "
            "pseudosequence match"
        )

high_error = high_error[
    high_error["pseudo_concat"].str.len() > 0
].copy()

low_error = low_error[
    low_error["pseudo_concat"].str.len() > 0
].copy()


# Calculate normalized pseudosequence similarity
def hamming_similarity(
    s1,
    s2
):

    length = min(
        len(s1),
        len(s2)
    )

    if length == 0:
        return np.nan

    matches = sum(
        a == b
        for a, b in zip(
            s1[:length],
            s2[:length]
        )
    )
    return matches / length


def pairwise_similarities(seqs):
    similarities = []
    for s1, s2 in combinations(
        seqs,
        2
    ):
        similarity = hamming_similarity(
            s1,
            s2
        )
        if not np.isnan(similarity):
            similarities.append(
                similarity
            )
    return similarities


# Compute within-group pairwise similarities
print("\nComputing within-group pairwise similarities...")

high_seqs = (high_error["pseudo_concat"].tolist())
low_seqs = (low_error["pseudo_concat"].tolist())
high_sims = pairwise_similarities(high_seqs)
low_sims = pairwise_similarities(low_seqs)
print(f"High-error pairs: {len(high_sims)}")
print(f"Low-error pairs: {len(low_sims)}")


print("Within-group similarity")
mean_high = np.mean(high_sims)
mean_low = np.mean(low_sims)
std_high = np.std(high_sims)
std_low = np.std(low_sims)

print(
    f"\nHigh-error group: "
    f"{mean_high:.4f} ± {std_high:.4f}"
)

print(
    f"Low-error group:  "
    f"{mean_low:.4f} ± {std_low:.4f}"
)

print(
    f"Difference:       "
    f"{mean_high - mean_low:.4f}"
)


print("\nComputing per-allotype mean similarity to all other allotypes...")
valid_ap = ap_df[
    ap_df["pseudo_concat"].str.len() > 0
].copy()

all_seqs = (valid_ap["pseudo_concat"].tolist())
all_allotypes = (valid_ap["allotype"].tolist())

per_allotype_sim = []

for i, seq_i in enumerate(all_seqs):

    similarities = []

    for j, seq_j in enumerate(all_seqs):

        if i == j:
            continue

        similarity = hamming_similarity(
            seq_i,
            seq_j
        )

        if not np.isnan(similarity):
            similarities.append(
                similarity
            )

    per_allotype_sim.append(
        np.mean(similarities)
        if similarities
        else np.nan
    )

valid_ap["mean_similarity_to_others"] = (
    per_allotype_sim
)


# Compare per-allotype similarity between performance groups
high_per_allotype = valid_ap[
    valid_ap["AP"] <= ap_thresh_low
]["mean_similarity_to_others"].dropna()

low_per_allotype = valid_ap[
    valid_ap["AP"] >= ap_thresh_high
]["mean_similarity_to_others"].dropna()

if (
    len(high_per_allotype) > 0
    and len(low_per_allotype) > 0
):

    stat, p_mw = mannwhitneyu(
        high_per_allotype,
        low_per_allotype,
        alternative="two-sided"
    )

else:

    stat = np.nan
    p_mw = np.nan


print("\nPer-allotype similarity comparison:")
print(f"High-error mean similarity: " f"{high_per_allotype.mean():.4f}")

print(f"Low-error mean similarity:  " f"{low_per_allotype.mean():.4f}")
print(f"Mann-Whitney U: {stat:.1f}")
print(f"p = {p_mw:.3e}")

# Correlation between sequence-space isolation and AP
valid_corr = valid_ap[["mean_similarity_to_others", "AP"]].dropna()
rho, p_rho = spearmanr(valid_corr["mean_similarity_to_others"], valid_corr["AP"])

print("\nSimilarity vs AP:")
print(f"Spearman rho = {rho:.4f}")
print(f"p = {p_rho:.3e}")
print(f"N = {len(valid_corr)}")

# Interpret similarity comparison
if p_mw < 0.05:
    if (
        high_per_allotype.mean()
        < low_per_allotype.mean()
    ):
        interp = (
            "High-error allotypes are significantly more "
            "sequence-isolated from the broader allotype set "
            "than low-error allotypes. This is consistent with "
            "poor performance being associated with unusual "
            "MHC sequence space rather than confusion between "
            "similar alleles."
        )
    else:
        interp = (
            "High-error allotypes are significantly more similar "
            "to the broader allotype set than low-error allotypes. "
            "This pattern is compatible with performance being "
            "influenced by local sequence similarity or allele "
            "confusion, rather than simple sequence-space isolation."
        )
else:
    interp = (
        "There is no significant difference in per-allotype "
        "sequence-space similarity between high- and low-error "
        "groups. Sequence similarity alone does not explain "
        "the observed performance differences."
    )
print(f"\nInterpretation: {interp}")

# Save per-allotype similarity table
out_cols = [
    "allotype",
    "gene",
    "n",
    "AP",
    "mean_similarity_to_others"
]

valid_ap[
    out_cols
].sort_values(
    "AP"
).to_csv(
    OUT_T / "mhc_similarity_error.csv",
    index=False
)

print(f"\nSaved: {OUT_T / 'mhc_similarity_error.csv'}")

# Plotting configuration
GENE_COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26",
    "OTHER": "#aaaaaa"
}

# Figure 1: Within-group pairwise similarity
fig, ax = plt.subplots(figsize=(7, 5))

bp = ax.boxplot(
    [
        high_sims,
        low_sims
    ],
    labels=[
        (
            "High-error\n"
            f"(bottom 20% AP)\n"
            f"n={len(high_error)} allotypes"
        ),
        (
            "Low-error\n"
            f"(top 20% AP)\n"
            f"n={len(low_error)} allotypes"
        )
    ],
    patch_artist=True,
    widths=0.4,
    medianprops={
        "color": "white",
        "linewidth": 2
    }
)

bp["boxes"][0].set_facecolor("#d6604d")
bp["boxes"][1].set_facecolor("#2166ac")

for element in [
    "whiskers",
    "caps",
    "fliers"
]:
    for item in bp[element]:
        item.set_color(
            "#555555"
        )

ax.scatter(
    [1, 2],
    [
        mean_high,
        mean_low
    ],
    color="white",
    s=50,
    zorder=5,
    marker="D",
    label="Mean"
)

ax.set_ylabel(
    "Pairwise pseudosequence similarity",
    fontsize=11
)
ax.set_title(
    "Within-group pseudosequence similarity\n"
    f"Δmean = {mean_high - mean_low:.4f}",
    fontsize=10
)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(OUT_F / "mhc_similarity_boxplot.png", dpi=300)

plt.close()
print("Saved: mhc_similarity_boxplot.png")

# Figure 2: Mean similarity to all others versus AP
fig, ax = plt.subplots(
    figsize=(8, 5)
)

for gene, color in GENE_COLORS.items():
    sub = valid_ap[
        valid_ap["gene"] == gene
    ]
    ax.scatter(
        sub["mean_similarity_to_others"],
        sub["AP"],
        label=f"HLA-{gene} (n={len(sub)})",
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=55
    )

# Overall linear trend
z = np.polyfit(
    valid_corr["mean_similarity_to_others"],
    valid_corr["AP"],
    1
)
x_line = np.linspace(
    valid_corr["mean_similarity_to_others"].min(),
    valid_corr["mean_similarity_to_others"].max(),
    100
)
ax.plot(
    x_line,
    np.poly1d(z)(x_line),
    "k--",
    linewidth=1.1,
    alpha=0.6,
    label="Linear trend"
)
ax.set_xlabel("Mean pseudosequence similarity to all other allotypes", fontsize=11)
ax.set_ylabel("Average Precision (AP)", fontsize=11)
ax.set_title(
    "MHC sequence-space similarity vs Graph-pMHC performance\n"
    f"Spearman ρ = {rho:.3f}, "
    f"p = {p_rho:.2e}, "
    f"N = {len(valid_corr)}",
    fontsize=10
)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_F / "mhc_similarity_vs_ap.png",dpi=300)
plt.close()
print("Saved: mhc_similarity_vs_ap.png")

# Final summary
print("SUMMARY")
print(f"\nHigh-error within-group similarity: " f"{mean_high:.4f}")
print(f"Low-error within-group similarity:   " f"{mean_low:.4f}")
print(f"Within-group difference:             " f"{mean_high - mean_low:.4f}")
print(f"\nHigh-error mean similarity to others:" f"{high_per_allotype.mean():.4f}")
print(f"Low-error mean similarity to others:   " f"{low_per_allotype.mean():.4f}")
print(f"Per-allotype Mann-Whitney p:           " f"{p_mw:.3e}")
print(f"Similarity-AP Spearman rho:            " f"{rho:.4f}")
print(f"Similarity-AP p:                       " f"{p_rho:.3e}")
print(f"\n{interp}")
print(f"\nOutputs saved to {OUT_T.parent}")
