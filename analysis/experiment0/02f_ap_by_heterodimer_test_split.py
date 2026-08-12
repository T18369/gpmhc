"""
02f_ap_by_heterodimer_test_split.py

Experiment 0.2f - Heterodimer-level AP evaluation

Evaluates Graph-pMHC and NetMHCIIpan at the individual MHC-II
heterodimer level. Only unambiguous alpha/beta pairs are retained:

    DP: DPA1 + DPB1
    DQ: DQA1 + DQB1
    DR: DRA + DRB

This avoids assigning a prediction to a single molecule when the
original allotype annotation contains multiple possible heterodimers.

NetMHCIIpan reports percentile rank, where lower values indicate
stronger predictions. Rank is therefore multiplied by -1 before AP
calculation so that higher scores indicate stronger predictions,
matching Graph-pMHC.

Input:
    Presentation_df_w_preds.csv

Outputs:
    results_heterodimer_test/
        tables/
            graph_ap_heterodimer.csv
            netmhcpan_ap_heterodimer.csv
            graph_vs_net_heterodimer.csv
        figures/
            graph_heterodimer_AP.png
            net_heterodimer_AP.png
            graph_vs_net_scatter_all.png
            graph_heterodimer_AP_distribution.png
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from matplotlib.patches import Patch


DATA = Path("Presentation_df_w_preds.csv")

OUT = Path("results_heterodimer_test")
OUT_T = OUT / "tables"
OUT_F = OUT / "figures"

OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 100

MOLECULE_COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26",
}


# Load the predefined test split.
df = pd.read_csv(
    DATA,
    usecols=[
        "EL",
        "Graph-pMHC_score",
        "NetMHCIIPan-4.0",
        "allotype",
        "split",
    ],
    low_memory=False,
)

df = df[df["split"] == "test"].copy()

print(f"TEST rows: {len(df):,}")


def extract_full_alleles(allotype):
    """Extract individual MHC-II allele names from an allotype string."""
    return [
        x.replace(" ", "")
        for x in re.findall(
            r"(?:DPA1|DPB1|DQA1|DQB1|DRA|DRB[1-9])\s*\*\s*\d+:\d+",
            str(allotype),
        )
    ]


def allele_gene(allele):
    return re.match(r"[A-Z0-9]+", allele).group()


def classify_heterodimer(allotype):
    """
    Return DP, DQ, or DR only when the annotation contains exactly
    one complete alpha/beta heterodimer.
    """
    alleles = extract_full_alleles(allotype)

    if len(alleles) != 2:
        return None

    genes = {allele_gene(allele) for allele in alleles}

    if genes == {"DPA1", "DPB1"}:
        return "DP"

    if genes == {"DQA1", "DQB1"}:
        return "DQ"

    if "DRA" in genes and any(g.startswith("DRB") for g in genes):
        return "DR"

    return None


def canonical_label(allotype):
    """Create a consistent label for the retained heterodimer."""
    return "__".join(extract_full_alleles(allotype))


# Keep only annotations that map cleanly to one peptide-binding heterodimer.
df["molecule"] = df["allotype"].apply(classify_heterodimer)
df = df[df["molecule"].notna()].copy()
df["heterodimer"] = df["allotype"].apply(canonical_label)

print(f"Heterodimer rows: {len(df):,}")
print(f"Unique heterodimers: {df['heterodimer'].nunique()}")


def calculate_ap(data, score_column):
    """Calculate AP independently for each heterodimer."""
    results = []

    for hla, group in data.groupby("heterodimer"):
        group = group.dropna(subset=["EL", score_column])

        if len(group) < MIN_SAMPLES:
            continue

        if group["EL"].nunique() < 2:
            continue

        results.append({
            "heterodimer": hla,
            "molecule": group["molecule"].iloc[0],
            "n": len(group),
            "n_pos": int(group["EL"].sum()),
            "AP": average_precision_score(
                group["EL"],
                group[score_column],
            ),
        })

    return (
        pd.DataFrame(results)
        .sort_values("AP")
        .reset_index(drop=True)
    )


print("\nComputing Graph-pMHC AP...")
graph_ap = calculate_ap(df, "Graph-pMHC_score")

graph_ap.to_csv(
    OUT_T / "graph_ap_heterodimer.csv",
    index=False,
)


# NetMHCIIpan percentile rank is lower for stronger predictions.
df["Net_score"] = -df["NetMHCIIPan-4.0"]

print("Computing NetMHCIIpan AP...")
net_ap = calculate_ap(df, "Net_score")

net_ap.to_csv(
    OUT_T / "netmhcpan_ap_heterodimer.csv",
    index=False,
)


# Compare only heterodimers for which both models have valid AP estimates.
comparison = graph_ap.merge(
    net_ap,
    on="heterodimer",
    suffixes=("_Graph", "_Net"),
)

comparison.to_csv(
    OUT_T / "graph_vs_net_heterodimer.csv",
    index=False,
)

print(f"\nShared heterodimers: {len(comparison)}")


legend_elements = [
    Patch(
        facecolor=color,
        label=f"HLA-{name}",
    )
    for name, color in MOLECULE_COLORS.items()
]


def ranked_bar(data, title, outfile):
    """Plot the lowest- and highest-performing heterodimers."""
    display = pd.concat(
        [data.head(20), data.tail(10)]
    ).drop_duplicates("heterodimer")

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.barh(
        range(len(display)),
        display["AP"],
        color=[
            MOLECULE_COLORS[x]
            for x in display["molecule"]
        ],
    )

    ax.set_yticks(range(len(display)))
    ax.set_yticklabels(display["heterodimer"], fontsize=8)
    ax.set_xlabel("Average Precision")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)

    ax.legend(
        handles=legend_elements,
        title="Molecule class",
    )

    plt.subplots_adjust(left=0.42)

    plt.savefig(
        OUT_F / outfile,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


ranked_bar(
    graph_ap,
    "Graph-pMHC heterodimer AP (TEST)",
    "graph_heterodimer_AP.png",
)

ranked_bar(
    net_ap,
    "NetMHCIIpan heterodimer AP (TEST)",
    "net_heterodimer_AP.png",
)


# Compare model performance across heterodimers with AP estimates from both models.
fig, ax = plt.subplots(figsize=(8, 8))

for molecule, group in comparison.groupby("molecule_Graph"):
    ax.scatter(
        group["AP_Net"],
        group["AP_Graph"],
        s=90,
        alpha=0.8,
        color=MOLECULE_COLORS[molecule],
        label=f"HLA-{molecule}",
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
)

r, p = pearsonr(
    comparison["AP_Net"],
    comparison["AP_Graph"],
)

ax.set_xlabel("NetMHCIIpan AP")
ax.set_ylabel("Graph-pMHC AP")
ax.set_title(f"Heterodimer AP comparison\nr = {r:.3f}")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.legend(
    handles=legend_elements,
    title="Molecule class",
)

plt.tight_layout()

plt.savefig(
    OUT_F / "graph_vs_net_scatter_all.png",
    dpi=300,
)

plt.close()


# Show the distribution of Graph-pMHC AP across evaluated heterodimers.
fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    graph_ap["AP"],
    bins=20,
    edgecolor="white",
)

ax.axvline(
    graph_ap["AP"].mean(),
    linestyle="--",
    color="gray",
    label=f"Mean = {graph_ap['AP'].mean():.3f}",
)

ax.set_xlabel("Average Precision")
ax.set_ylabel("Number of heterodimers")

ax.set_title(
    "Distribution of Graph-pMHC heterodimer AP (TEST)\n"
    f"Mean = {graph_ap['AP'].mean():.3f}   "
    f"Range = {graph_ap['AP'].min():.3f}-{graph_ap['AP'].max():.3f}"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    OUT_F / "graph_heterodimer_AP_distribution.png",
    dpi=300,
)

plt.close()


print("\nLowest Graph-pMHC heterodimers:")
print(
    graph_ap.head(20).to_string(index=False)
)

print(f"\nSaved: {OUT}")
