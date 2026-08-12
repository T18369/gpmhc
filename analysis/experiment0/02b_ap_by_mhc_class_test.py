#!/usr/bin/env python3

"""
Calculate Graph-pMHC and NetMHCIIpan AP on the TEST split
for DP, DQ, DR, and DP+DR excluding DQ.
"""

import pandas as pd
from sklearn.metrics import average_precision_score


INPUT = "Presentation_df_w_preds.csv"
OUTPUT = "ap_by_mhc_class_test.csv"


# Load predictions and keep the predefined test split.
df = pd.read_csv(
    INPUT,
    low_memory=False
)

print("Loaded:", df.shape)

df = df[
    df["split"].astype(str).str.lower() == "test"
].copy()

print("Test split:", df.shape)


def has_class(molecule, cls):
    """
    Check whether a molecule contains the expected
    alpha/beta chains for an MHC-II class.
    """

    molecule = str(molecule).upper()

    if cls == "DP":
        return "DPA1" in molecule and "DPB1" in molecule

    if cls == "DQ":
        return "DQA1" in molecule and "DQB1" in molecule

    if cls == "DR":
        return "DRA" in molecule and "DRB" in molecule

    return False


df["has_DP"] = df["molecule"].apply(
    lambda x: has_class(x, "DP")
)

df["has_DQ"] = df["molecule"].apply(
    lambda x: has_class(x, "DQ")
)

df["has_DR"] = df["molecule"].apply(
    lambda x: has_class(x, "DR")
)


print("\nClass counts")
print("DP:", df["has_DP"].sum())
print("DQ:", df["has_DQ"].sum())
print("DR:", df["has_DR"].sum())


def calculate_ap(subset, name):

    # Graph-pMHC scores are already oriented so that
    # larger values correspond to stronger predictions.
    graph = subset.dropna(
        subset=["EL", "Graph-pMHC_score"]
    )

    graph_ap = average_precision_score(
        graph["EL"],
        graph["Graph-pMHC_score"]
    )

    # NetMHCIIpan reports percentile rank, where lower
    # values indicate stronger predicted binding. Reverse
    # the sign so both predictors use the same orientation.
    net = subset.dropna(
        subset=["EL", "NetMHCIIPan-4.0"]
    ).copy()

    net["Net_score"] = -net["NetMHCIIPan-4.0"]

    net_ap = average_precision_score(
        net["EL"],
        net["Net_score"]
    )

    return {
        "comparison": name,
        "Graph_n": len(graph),
        "NetMHC_n": len(net),
        "positive_rate": graph["EL"].mean(),
        "Graph-pMHC_AP": graph_ap,
        "NetMHCIIpan_AP": net_ap,
        "AP_difference": graph_ap - net_ap,
        "relative_AP_improvement_percent": (
            (graph_ap - net_ap) / net_ap * 100
        )
    }


comparisons = {
    "DP": df[df["has_DP"]],
    "DQ": df[df["has_DQ"]],
    "DR": df[df["has_DR"]],
    "DP+DR (DQ excluded)": df[
        (df["has_DP"] | df["has_DR"])
        & ~df["has_DQ"]
    ]
}


results = []

for name, subset in comparisons.items():

    print(f"\n{name}: {len(subset):,} examples")

    results.append(
        calculate_ap(
            subset,
            name
        )
    )


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT,
    index=False
)


print("\nResults")
print("-" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:", OUTPUT)
