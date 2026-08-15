#!/usr/bin/env python3
"""
Characterize peptide-MHC contact architecture

Contact definitions:
    Hard       <= 4 A
    Proximal   >4–8 A
    Distal     >8 A

Inputs:
    experiment1/distance_maps/continuous/*.csv
    results/tables/ap_by_allotype.csv
Outputs:
    experiment1/figures/contact_architecture_summary.png
    experiment1/figures/contact_ratio_vs_ap.png
    experiment1/tables/contact_architecture_allotype.csv
    experiment1/tables/contact_ratio_vs_ap.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DIST_DIR = Path("experiment1/distance_maps")
AP_FILE = Path("results/tables/ap_by_allotype.csv")

OUT_F = Path("experiment1/figures")
OUT_T = Path("experiment1/tables")
OUT_F.mkdir(parents=True, exist_ok=True)
OUT_T.mkdir(parents=True, exist_ok=True)

HARD_CUTOFF = 4.0
PROXIMAL_CUTOFF = 8.0

GENE_COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26",
}

def normalise_allotype(value):
    value = str(value).strip()
    value = value.replace("____", "___")
    value = value.replace("__", "___")
    return value

def classify_gene(allotype):
    value = str(allotype).upper()
    if "DRB" in value:
        return "DR"
    if "DPB" in value:
        return "DP"
    if "DQB" in value:
        return "DQ"
    return "Unknown"

def contact_category(distance):
    if pd.isna(distance):
        return None
    if distance <= HARD_CUTOFF:
        return "hard"
    if distance <= PROXIMAL_CUTOFF:
        return "proximal"
    return "distal"

if not DIST_DIR.exists():
    raise SystemExit(
        f"Missing distance-map directory: {DIST_DIR}\n"
        "Run 03_extract_distance_maps.py first."
    )

map_files = sorted(
    p for p in DIST_DIR.glob("*.csv")
    if p.name not in {
        "structural_metrics.csv",
        "allotype_metrics.csv",
        "extraction_qc.csv",
        "extraction_failures.csv",
    }
)

if not map_files:
    raise SystemExit(f"No distance-map CSV files found in {DIST_DIR}")

print("Contact architecture analysis")
print(f"Distance maps : {len(map_files)}")
print(
    f"Distance cutoffs : ≤{HARD_CUTOFF:.0f} Å hard | "
    f"{HARD_CUTOFF:.0f}–{PROXIMAL_CUTOFF:.0f} Å proximal | "
    f">{PROXIMAL_CUTOFF:.0f} Å distal"
)

frames = []
for path in map_files:
    try:
        df = pd.read_csv(path)
        required = {
            "allotype",
            "distance_A",
            "mhc_chain_type",
        }
        missing = required - set(df.columns)
        if missing:
            print(
                f"SKIP {path.name}: "
                f"missing {', '.join(sorted(missing))}"
            )
            continue
        frames.append(df)
    except Exception as exc:
        print(f"SKIP {path.name}: {str(exc)}")

if not frames:
    raise SystemExit("No usable distance maps were found.")

distances = pd.concat(frames, ignore_index=True,)
distances["allotype_norm"] = (distances["allotype"].apply(normalise_allotype))
distances["gene"] = (distances["allotype"].apply(classify_gene))
distances["contact_class"] = (distances["distance_A"].apply(contact_category))
distances = distances[distances["distance_A"].notna()].copy()
print(f"Distance pairs loaded : {len(distances):,}")
print(f"Allotypes represented  : " f"{distances['allotype_norm'].nunique()}")

def summarize_subset(df, label):
    total = len(df)
    hard = int((df["distance_A"] <= HARD_CUTOFF).sum())
    proximal = int(
        (
            (df["distance_A"] > HARD_CUTOFF)
            & (df["distance_A"] <= PROXIMAL_CUTOFF)
        ).sum()
    )
    distal = int((df["distance_A"] > PROXIMAL_CUTOFF).sum())
    within_8 = hard + proximal
    ratio = (
        proximal / hard
        if hard > 0
        else np.inf
    )
    return {
        "region": label,
        "n_pairs": total,
        "hard_contact_count": hard,
        "proximal_count": proximal,
        "distal_count": distal,
        "within_8A_count": within_8,
        "soft_to_hard_ratio": ratio,
    }

rows = []
for allotype, group in distances.groupby(
    ["allotype_norm", "gene"],
    sort=True,
):
    allotype_name, gene = allotype
    combined = summarize_subset(group, "combined",)
    alpha = summarize_subset(
        group[
            group["mhc_chain_type"]
            .str.lower()
            == "alpha"
        ],
        "alpha",
    )
    beta = summarize_subset(
        group[
            group["mhc_chain_type"]
            .str.lower()
            == "beta"
        ],
        "beta",
    )
    row = {
        "allotype": allotype_name,
        "gene": gene,
    }
    for summary in (
        combined,
        alpha,
        beta,
    ):
        prefix = summary["region"]
        for key, value in summary.items():
            if key == "region":
                continue
            row[
                f"{prefix}_{key}"
            ] = value
    rows.append(row)
contact_df = pd.DataFrame(rows)

if not AP_FILE.exists():
    print(f"\nAP file not found: {AP_FILE}")
    ap_df = None
else:
    ap_df = pd.read_csv(AP_FILE)
    required_ap = {"allotype","AP",}
    missing_ap = (required_ap - set(ap_df.columns))

    if missing_ap:
        print("AP file is missing: " + ", ".join(sorted(missing_ap)))
        ap_df = None
    else:
        ap_df["allotype_norm"] = (ap_df["allotype"].apply(normalise_allotype))
if ap_df is not None:
    contact_df = contact_df.merge(
        ap_df[["allotype_norm","AP",]],
        left_on="allotype",
        right_on="allotype_norm",
        how="left",
    )
    contact_df.drop(columns=["allotype_norm"], inplace=True,)

contact_table = (OUT_T / "contact_architecture_allotype.csv")
contact_df.to_csv(contact_table, index=False,)

print("Contact architecture")
display_columns = [
    "allotype",
    "gene",
    "combined_hard_contact_count",
    "combined_proximal_count",
    "combined_distal_count",
    "combined_soft_to_hard_ratio",
    "alpha_hard_contact_count",
    "alpha_proximal_count",
    "beta_hard_contact_count",
    "beta_proximal_count",
]

available = [
    c
    for c in display_columns
    if c in contact_df.columns
]

print(contact_df[available].to_string(index=False))
print(f"Saved: {contact_table}")
plot_df = contact_df.copy()
x = np.arange(len(plot_df))
labels = [str(a).replace("___", "\n") for a in plot_df["allotype"]]
fig, axes = plt.subplots(1, 2,figsize=(15, 6),)

# Panel A: combined MHC contact classes
ax = axes[0]
width = 0.25
hard_values = (plot_df["combined_hard_contact_count"])
proximal_values = (plot_df["combined_proximal_count"])
distal_values = (plot_df["combined_distal_count"])

ax.bar(x - width, hard_values, width, label="Hard ≤4 Å", alpha=0.9,)
ax.bar(x, proximal_values, width, label="Proximal 4–8 Å", alpha=0.65,)
ax.bar(x + width, distal_values, width, label="Distal >8 Å", alpha=0.35,)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8,)
ax.set_ylabel("Peptide–MHC residue pairs")
ax.set_title("Combined MHC contact architecture")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3,)

# Panel B: alpha vs beta hard/proximal contacts
ax = axes[1]
alpha_hard = (plot_df["alpha_hard_contact_count"])
alpha_proximal = (plot_df["alpha_proximal_count"])
beta_hard = (plot_df["beta_hard_contact_count"])
beta_proximal = (plot_df["beta_proximal_count"])
offset = 0.18

ax.bar(x - 1.5 * offset, alpha_hard, width=offset, label="α hard", alpha=0.9,)
ax.bar(x - 0.5 * offset, alpha_proximal, width=offset, label="α proximal", alpha=0.55,)
ax.bar(x + 0.5 * offset, beta_hard, width=offset, label="β hard", alpha=0.9,)
ax.bar(x + 1.5 * offset, beta_proximal, width=offset, label="β proximal", alpha=0.55,)
ax.set_xticks(x)
ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right",
    fontsize=8,
)
ax.set_ylabel("Peptide–MHC residue pairs")
ax.set_title("Alpha vs beta contact architecture")
ax.legend(fontsize=8, ncol=2,)
ax.grid(axis="y", alpha=0.3,)
fig.suptitle("Peptide–MHC contact architecture", fontsize=12,)
plt.tight_layout()
summary_figure = (OUT_F / "contact_architecture_summary.png")
plt.savefig(summary_figure, dpi=300, bbox_inches="tight",)
plt.close()
print(f"Saved: {summary_figure}")

print("Contact ratio vs model performance")
if "AP" not in contact_df.columns:
    print("AP values unavailable. " "Skipping correlation.")
else:
    correlation_df = contact_df[
        [
            "allotype",
            "gene",
            "AP",
            "combined_soft_to_hard_ratio",
        ]
    ].copy()
    correlation_df = (
        correlation_df[
            np.isfinite(
                correlation_df["combined_soft_to_hard_ratio"]
            )
        ]
        .dropna(subset=["AP", "combined_soft_to_hard_ratio",])
    )

    correlation_df.to_csv(OUT_T / "contact_ratio_vs_ap.csv", index=False,)

    if len(correlation_df) >= 3:
        rho, p = spearmanr(
            correlation_df["combined_soft_to_hard_ratio"],
            correlation_df["AP"],
        )
        print(f"Spearman ρ = {rho:.4f}")
        print(f"p = {p:.3e}")
        print(f"N = {len(correlation_df)}")

        fig, ax = plt.subplots(figsize=(7, 5))

        for gene, color in GENE_COLORS.items():
            sub = correlation_df[
                correlation_df["gene"]
                == gene
            ]
            if sub.empty:
                continue
            ax.scatter(
                sub["combined_soft_to_hard_ratio"],
                sub["AP"],
                color=color,
                s=80,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=f"HLA-{gene} "
                      f"(n={len(sub)})",
            )
            for _, row in sub.iterrows():
                label = (row["allotype"].split("___")[-1])
                ax.annotate(
                    label,
                    (
                        row["combined_soft_to_hard_ratio"],
                        row["AP"],
                    ),
                    fontsize=7,
                    xytext=(4, 4),
                    textcoords="offset points",
                )

        if (
            correlation_df["combined_soft_to_hard_ratio"].nunique()
            >= 2
        ):
            x_values = (correlation_df["combined_soft_to_hard_ratio"])
            y_values = (correlation_df["AP"])
            coefficients = np.polyfit(
                x_values,
                y_values,
                1,
            )
            x_line = np.linspace(
                x_values.min(),
                x_values.max(),
                100,
            )
            ax.plot(
                x_line,
                np.poly1d(coefficients)(x_line),
                "k--",
                linewidth=1,
                alpha=0.6,
            )
        ax.axvline(
            1.0,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
        ax.set_xlabel("Proximal / hard contact ratio")
        ax.set_ylabel("Graph-pMHC AP")
        ax.set_ylim(0, 1.05,)
        ax.set_title(
            "Contact ratio vs Graph-pMHC performance\n"
            f"Spearman ρ = {rho:.3f}, "
            f"p = {p:.2e}, N = {len(correlation_df)}"
        )
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        ratio_figure = (OUT_F / "contact_ratio_vs_ap.png")
        plt.savefig(
            ratio_figure,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Saved: {ratio_figure}")
    else:
        print(f"Only {len(correlation_df)} ""finite AP/allotype pairs. "
            "Need ≥3 for correlation."
        )
print("Analysis complete.")
