#!/usr/bin/env python3

"""
Calculate Graph-pMHC and NetMHCIIpan performance at the
individual HLA-II allele level.

The original dataset contains composite allotype strings, for example:

    DQA1*05:01___DQB1*03:02
    DRA*01:01___DRB5*01:01

and, in some cases, multi-locus combinations.

This analysis expands each observation into its constituent
HLA-II alleles and calculates AP independently for each allele.

Retained genes:
    DPA1, DPB1
    DQA1, DQB1
    DRA, DRB1, DRB3, DRB4, DRB5

Important:
    An observation containing multiple alleles contributes to
    each constituent allele. The allele-level analysis therefore
    describes performance associated with each allele, rather
    than creating independent peptide-level observations.

NetMHCIIpan reports percentile rank, where lower rank indicates
stronger predicted binding. The rank is multiplied by -1 before
calculating AP so that both models use the same score direction.

Input:
    Presentation_df_w_preds.csv

Outputs:
    results_individual_allele_test/
        tables/
            graph_ap_by_allele.csv
            netmhcpan_ap_by_allele.csv
            graph_vs_net_ap_by_allele.csv

        figures/
            graph_ap_by_allele.png
            netmhcpan_ap_by_allele.png
            ap_distribution_by_gene.png
            graph_vs_net_scatter.png
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import average_precision_score
from matplotlib.patches import Patch


DATA = Path("Presentation_df_w_preds.csv")

OUT = Path("results_individual_allele_test")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES = 100


# Load the predefined TEST split.
df = pd.read_csv(
    DATA,
    low_memory=False
)

df = df[
    df["split"].astype(str).str.lower() == "test"
].copy()

print("=" * 70)
print("Loading TEST split")
print("=" * 70)
print(f"Rows: {len(df):,}")


# Extract the individual HLA-II alleles contained in each
# composite allotype string.
ALLELE_REGEX = re.compile(
    r"(?:DPA1|DPB1|DQA1|DQB1|DRA|DRB1|DRB3|DRB4|DRB5)"
    r"\*\d{2}:\d{2}"
)


def extract_alleles(allotype):

    if pd.isna(allotype):
        return []

    return list(
        set(
            ALLELE_REGEX.findall(
                str(allotype)
            )
        )
    )


# Expand the dataset so each constituent allele receives the peptide, label, and model predictions from the
# original allotype-level observation.
expanded = []

for _, row in df.iterrows():

    for allele in extract_alleles(row["allotype"]):

        expanded.append({
            "allele": allele,
            "gene": allele.split("*")[0],
            "EL": row["EL"],
            "Graph-pMHC_score": row["Graph-pMHC_score"],
            "NetMHCIIPan_rank": row["NetMHCIIPan-4.0"]
        })


allele_df = pd.DataFrame(expanded)

print(f"Expanded rows: {len(allele_df):,}")
print(f"Individual alleles: {allele_df['allele'].nunique()}")


# NetMHCIIpan uses percentile rank:
# lower rank = stronger prediction.
# Reverse the sign so higher values correspond to stronger predictions, matching Graph-pMHC.
allele_df["NetMHCIIPan_score"] = (
    -allele_df["NetMHCIIPan_rank"]
)


def compute_ap(score_column):

    results = []

    for allele, group in allele_df.groupby("allele"):

        group = group.dropna(
            subset=[
                "EL",
                score_column
            ]
        )

        if len(group) < MIN_SAMPLES:
            continue

        if group["EL"].nunique() < 2:
            continue

        results.append({
            "allele": allele,
            "gene": allele.split("*")[0],
            "n": len(group),
            "n_pos": int(group["EL"].sum()),
            "n_neg": int((1 - group["EL"]).sum()),
            "positive_fraction": group["EL"].mean(),
            "AP": average_precision_score(
                group["EL"],
                group[score_column]
            )
        })

    return (
        pd.DataFrame(results)
        .sort_values("AP")
        .reset_index(drop=True)
    )


# Graph-pMHC
print("\nCalculating Graph-pMHC AP...")

graph_ap = compute_ap(
    "Graph-pMHC_score"
)

graph_ap.to_csv(
    TABLES / "graph_ap_by_allele.csv",
    index=False
)


# NetMHCIIpan
print("Calculating NetMHCIIpan AP...")

net_ap = compute_ap(
    "NetMHCIIPan_score"
)

net_ap.to_csv(
    TABLES / "netmhcpan_ap_by_allele.csv",
    index=False
)


# Compare the two predictors only for alleles with
# valid estimates from both analyses.
comparison = graph_ap.merge(
    net_ap,
    on="allele",
    suffixes=(
        "_Graph",
        "_Net"
    )
)

comparison["AP_difference"] = (
    comparison["AP_Graph"]
    - comparison["AP_Net"]
)

comparison.to_csv(
    TABLES / "graph_vs_net_ap_by_allele.csv",
    index=False
)


# Plot the lowest- and highest-performing alleles.
def ranked_barplot(data, title, outfile):

    display = pd.concat([
        data.head(20),
        data.tail(10)
    ]).drop_duplicates("allele")

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.barh(
        range(len(display)),
        display["AP"],
        edgecolor="white"
    )

    ax.set_yticks(
        range(len(display))
    )

    ax.set_yticklabels(
        display["allele"],
        fontsize=8
    )

    ax.set_xlabel(
        "Average Precision"
    )

    ax.set_xlim(
        0,
        1.05
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


ranked_barplot(
    graph_ap,
    "Graph-pMHC TEST split\nIndividual allele AP",
    FIGURES / "graph_ap_by_allele.png"
)

ranked_barplot(
    net_ap,
    "NetMHCIIpan TEST split\nIndividual allele AP",
    FIGURES / "netmhcpan_ap_by_allele.png"
)


# Direct comparison of allele-level performance.
fig, ax = plt.subplots(
    figsize=(7, 7)
)

ax.scatter(
    comparison["AP_Net"],
    comparison["AP_Graph"],
    alpha=0.75,
    s=35
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray"
)

ax.set_xlabel(
    "NetMHCIIpan AP"
)

ax.set_ylabel(
    "Graph-pMHC AP"
)

ax.set_title(
    "Individual allele AP comparison"
)

plt.tight_layout()

plt.savefig(
    FIGURES / "graph_vs_net_scatter.png",
    dpi=300
)

plt.close()


# Show the distribution of Graph-pMHC AP across individual
# allele classes.
fig, ax = plt.subplots(
    figsize=(10, 5)
)

gene_order = [
    "DPA1",
    "DPB1",
    "DQA1",
    "DQB1",
    "DRA",
    "DRB1",
    "DRB3",
    "DRB4",
    "DRB5"
]

groups = [
    graph_ap.loc[
        graph_ap["gene"] == gene,
        "AP"
    ].values
    for gene in gene_order
]

groups = [
    group
    for group in groups
    if len(group) > 0
]

labels = [
    gene
    for gene in gene_order
    if len(
        graph_ap.loc[
            graph_ap["gene"] == gene
        ]
    ) > 0
]

ax.boxplot(
    groups,
    labels=labels,
    showfliers=True
)

ax.set_ylabel(
    "Graph-pMHC AP"
)

ax.set_title(
    "Graph-pMHC AP distribution by allele"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    FIGURES / "ap_distribution_by_gene.png",
    dpi=300
)

plt.close()


print("\nLowest Graph-pMHC alleles")
print("-" * 50)

print(
    graph_ap[
        [
            "allele",
            "gene",
            "n",
            "n_pos",
            "AP"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\nSaved:")
print(OUT)
