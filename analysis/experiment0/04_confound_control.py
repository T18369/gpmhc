"""
04_confound_control.py

Experiment 0.4 — Controlling for Training Density
==================================================

The divergence-AP analysis in Experiment 0.3 found a negative
relationship between pseudosequence divergence and Graph-pMHC
performance. The key concern is whether this reflects a true
representation limitation or simply reduced training coverage.

Two competing explanations are therefore being tested:

    Divergent allele → fewer training examples → lower AP

versus:

    Divergent allele → structurally unusual → current representation
    cannot adequately capture its peptide-binding behavior

This script tests whether divergence remains associated with AP after
controlling for the number of training examples per allotype.

Four complementary analyses are performed:

    1. Training coverage vs AP
       Tests whether training density itself predicts performance.

    2. Partial correlation
       Tests whether divergence remains associated with AP after
       removing the linear effect of training coverage.

    3. Coverage-stratified analysis
       Tests the divergence-AP relationship within training-coverage
       quartiles, providing a non-parametric check against a coverage
       artifact.

    4. Multiple regression
       Models AP as a function of divergence and log10(training count)
       to quantify their independent contributions.

Interpretation:
    If divergence remains negatively associated with AP after these
    controls, the representation bottleneck hypothesis is strengthened.

Input:
    results/divergence/tables/divergence_vs_ap.csv

Required columns:
    allotype
    gene
    n
    AP
    consensus_divergence

Outputs:
    results/confound/tables/confound_control_summary.csv
    results/confound/figures/confound_control_coverage.png
    results/confound/figures/confound_control_partial.png
    results/confound/figures/confound_control_stratified.png
    results/confound/figures/confound_control_regression.png

Usage:
    python analysis/04_confound_control.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import linregress, pearsonr, spearmanr, t as t_dist
from matplotlib.patches import Patch
from numpy.linalg import inv, lstsq

import statsmodels.api as sm


# Paths and output directories
INPUT = Path(
    "results/divergence/tables/divergence_vs_ap.csv" #subfolder created in 03
)

OUT_ROOT = Path(
    "results/confound"
)

OUT_T = OUT_ROOT / "tables"
OUT_F = OUT_ROOT / "figures"

OUT_T.mkdir(
    parents=True,
    exist_ok=True
)

OUT_F.mkdir(
    parents=True,
    exist_ok=True
)


# Shared plotting colors
GENE_COLORS = {
    "DR": "#2166ac",
    "DP": "#d6604d",
    "DQ": "#4dac26"
}


# Load divergence and AP results
print("=" * 70)
print("Loading divergence and AP results")
print("=" * 70)

df = pd.read_csv(
    INPUT
)

df = df.dropna(
    subset=[
        "AP",
        "consensus_divergence",
        "n"
    ]
).copy()

df["log_n"] = np.log10(
    df["n"].clip(lower=1)
)

print(
    f"Allotypes: {len(df)}"
)

print(
    f"AP range: {df['AP'].min():.4f} – {df['AP'].max():.4f}"
)

print(
    f"Divergence range: "
    f"{df['consensus_divergence'].min():.4f} – "
    f"{df['consensus_divergence'].max():.4f}"
)

print(
    f"Training-count range: "
    f"{df['n'].min()} – {df['n'].max()}"
)


summary_rows = []


# Analysis 1: Does training coverage predict AP?
print("\n" + "=" * 60)
print("Analysis 1: Training coverage vs AP")
print("=" * 60)

rho_n, p_n = spearmanr(
    df["log_n"],
    df["AP"]
)

slope, intercept, r_value, p_value_lr, stderr = linregress(
    df["log_n"],
    df["AP"]
)

r_squared = r_value ** 2

print(
    f"Spearman rho(log10(n_train), AP) = "
    f"{rho_n:.4f}, p = {p_n:.3e}"
)

print(
    f"Linear slope = {slope:.4f}, "
    f"p = {p_value_lr:.3e}, "
    f"R² = {r_squared:.4f}"
)

summary_rows.append(
    {
        "analysis": "Coverage vs AP",
        "metric_x": "log10(n_train)",
        "metric_y": "AP",
        "rho_or_beta": rho_n,
        "p_value": p_n,
        "n": len(df),
        "interpretation": (
            "Coverage predicts AP"
            if p_n < 0.05
            else
            "Coverage does not significantly predict AP"
        )
    }
)


# Plot training coverage against AP
fig, ax = plt.subplots(
    figsize=(7, 5)
)

for gene, color in GENE_COLORS.items():

    sub = df[
        df["gene"] == gene
    ]

    ax.scatter(
        sub["log_n"],
        sub["AP"],
        label=f"HLA-{gene}",
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=55
    )


x_line = np.linspace(
    df["log_n"].min(),
    df["log_n"].max(),
    100
)

ax.plot(
    x_line,
    intercept + slope * x_line,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    alpha=0.8,
    label="Linear regression"
)


# LOESS provides a non-linear visual check
loess = sm.nonparametric.lowess(
    endog=df["AP"],
    exog=df["log_n"],
    frac=0.8,
    it=3,
    return_sorted=True
)

ax.plot(
    loess[:, 0],
    loess[:, 1],
    color="black",
    linewidth=2,
    label="LOESS"
)

ax.text(
    0.03,
    0.05,
    f"Spearman ρ = {rho_n:.3f} (p={p_n:.2e})\n"
    f"Linear β = {slope:.3f} (p={p_value_lr:.2e})\n"
    f"R² = {r_squared:.3f}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.8
    )
)

ax.set_xlabel(
    "log₁₀(training samples per allotype)",
    fontsize=11
)

ax.set_ylabel(
    "Average Precision (AP)",
    fontsize=11
)

ax.set_title(
    "Training coverage vs Graph-pMHC performance",
    fontsize=10
)

ax.set_ylim(
    0,
    1.05
)

ax.legend(
    fontsize=9
)

ax.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confound_control_coverage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: confound_control_coverage.png"
)


# Analysis 2: Partial correlation controlling for training coverage
print("\n" + "=" * 60)
print("Analysis 2: Partial correlation")
print("=" * 60)


def partial_corr(x, y, z):
    """
    Partial Pearson correlation between x and y while controlling
    for z using residualization.

    Returns:
        partial_r
        p_value
        n
        residual_x
        residual_y
    """

    data = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z
        }
    ).dropna()

    n = len(data)

    if n < 3:
        return np.nan, np.nan, n, np.array([]), np.array([])

    z_matrix = np.column_stack(
        [
            np.ones(n),
            data["z"].values
        ]
    )

    coef_x = lstsq(
        z_matrix,
        data["x"].values,
        rcond=None
    )[0]

    coef_y = lstsq(
        z_matrix,
        data["y"].values,
        rcond=None
    )[0]

    resid_x = (
        data["x"].values
        -
        z_matrix @ coef_x
    )

    resid_y = (
        data["y"].values
        -
        z_matrix @ coef_y
    )

    r, p = pearsonr(
        resid_x,
        resid_y
    )

    return (
        r,
        p,
        n,
        resid_x,
        resid_y
    )


r_partial, p_partial, n_partial, resid_div, resid_ap = partial_corr(
    df["consensus_divergence"],
    df["AP"],
    df["log_n"]
)

print(
    f"Partial r(divergence, AP | log10(n_train)) = "
    f"{r_partial:.4f}"
)

print(
    f"p = {p_partial:.3e}"
)

print(
    f"n = {n_partial}"
)


if p_partial < 0.05 and r_partial < 0:

    interp = (
        "Divergence remains negatively associated with AP after "
        "controlling for training coverage."
    )

elif p_partial < 0.05:

    interp = (
        "Divergence remains associated with AP after controlling "
        "for training coverage, but the direction is positive."
    )

else:

    interp = (
        "The divergence-AP relationship is not significant after "
        "controlling for training coverage."
    )


print(
    f"\nInterpretation: {interp}"
)

summary_rows.append(
    {
        "analysis": "Partial correlation",
        "metric_x": "consensus_divergence | log10(n_train)",
        "metric_y": "AP | log10(n_train)",
        "rho_or_beta": r_partial,
        "p_value": p_partial,
        "n": n_partial,
        "interpretation": interp
    }
)


# Plot residualized divergence against residualized AP
fig, ax = plt.subplots(
    figsize=(7, 5)
)

for gene, color in GENE_COLORS.items():

    mask = (
        df["gene"] == gene
    )

    ax.scatter(
        resid_div[mask],
        resid_ap[mask],
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.4,
        s=55,
        label=f"HLA-{gene}"
    )


if len(resid_div) >= 3:

    slope_partial, intercept_partial = np.polyfit(
        resid_div,
        resid_ap,
        1
    )

    x_partial = np.linspace(
        resid_div.min(),
        resid_div.max(),
        100
    )

    ax.plot(
        x_partial,
        intercept_partial + slope_partial * x_partial,
        "k--",
        linewidth=1.1,
        alpha=0.6
    )


ax.axhline(
    0,
    color="gray",
    linewidth=0.6,
    alpha=0.5
)

ax.axvline(
    0,
    color="gray",
    linewidth=0.6,
    alpha=0.5
)

ax.legend(
    fontsize=9
)

ax.set_xlabel(
    "Residual divergence after controlling for training coverage",
    fontsize=10
)

ax.set_ylabel(
    "Residual AP after controlling for training coverage",
    fontsize=10
)

ax.set_title(
    f"Partial correlation: divergence vs AP | log10(n_train)\n"
    f"r = {r_partial:.3f}, p = {p_partial:.2e}, N = {n_partial}",
    fontsize=10
)

ax.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confound_control_partial.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: confound_control_partial.png"
)


# Analysis 3: Does the divergence effect persist across coverage strata?
print("\n" + "=" * 60)
print("Analysis 3: Coverage-stratified divergence-AP correlation")
print("=" * 60)

df["coverage_quartile"] = pd.qcut(
    df["log_n"],
    q=4,
    labels=[
        "Q1 (lowest)",
        "Q2",
        "Q3",
        "Q4 (highest)"
    ]
)


strat_rows = []


for quartile, sub in df.groupby(
    "coverage_quartile",
    observed=True
):

    if len(sub) < 5:

        print(
            f"{quartile}: n={len(sub)} "
            "too few for correlation"
        )

        continue


    rho_q, p_q = spearmanr(
        sub["consensus_divergence"],
        sub["AP"]
    )

    n_q = len(sub)

    print(
        f"{quartile}: "
        f"n={n_q}, "
        f"rho={rho_q:.4f}, "
        f"p={p_q:.3e}"
    )

    strat_rows.append(
        {
            "quartile": str(quartile),
            "n": n_q,
            "rho": rho_q,
            "p": p_q,
            "n_range": (
                f"{sub['n'].min()}–"
                f"{sub['n'].max()}"
            )
        }
    )

    summary_rows.append(
        {
            "analysis": f"Coverage-stratified correlation ({quartile})",
            "metric_x": "consensus_divergence",
            "metric_y": "AP",
            "rho_or_beta": rho_q,
            "p_value": p_q,
            "n": n_q,
            "interpretation": (
                "Significant divergence-AP association"
                if p_q < 0.05
                else
                "No significant divergence-AP association"
            )
        }
    )


strat_df = pd.DataFrame(
    strat_rows
)


# Plot coverage-stratified correlations
quartiles = (
    df["coverage_quartile"]
    .cat.categories
)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharey=True
)

axes = axes.flatten()


for ax, quartile in zip(
    axes,
    quartiles
):

    sub = df[
        df["coverage_quartile"] == quartile
    ]

    if len(sub) < 3:

        ax.set_visible(False)

        continue


    for gene, color in GENE_COLORS.items():

        sg = sub[
            sub["gene"] == gene
        ]

        ax.scatter(
            sg["consensus_divergence"],
            sg["AP"],
            color=color,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            s=50,
            label=f"HLA-{gene}"
        )


    if len(sub) >= 4:

        slope_q, intercept_q = np.polyfit(
            sub["consensus_divergence"],
            sub["AP"],
            1
        )

        x_q = np.linspace(
            sub["consensus_divergence"].min(),
            sub["consensus_divergence"].max(),
            100
        )

        ax.plot(
            x_q,
            intercept_q + slope_q * x_q,
            "k--",
            linewidth=1,
            alpha=0.6
        )


    row = strat_df[
        strat_df["quartile"] == str(quartile)
    ]

    if len(row):

        rho_txt = (
            f"ρ={row['rho'].iloc[0]:.3f}, "
            f"p={row['p'].iloc[0]:.2e}"
        )

    else:

        rho_txt = ""


    ax.set_title(
        f"{quartile}\n{rho_txt}",
        fontsize=9
    )

    ax.set_xlabel(
        "Consensus divergence",
        fontsize=9
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )


axes[0].set_ylabel(
    "Average Precision (AP)",
    fontsize=10
)

axes[2].set_ylabel(
    "Average Precision (AP)",
    fontsize=10
)

axes[0].legend(
    fontsize=7
)

fig.suptitle(
    "Divergence-AP relationship stratified by training coverage",
    fontsize=10
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    OUT_F / "confound_control_stratified.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: confound_control_stratified.png"
)


# Analysis 4: Multiple regression AP ~ divergence + log10(training count)
print("\n" + "=" * 60)
print("Analysis 4: Multiple regression")
print("=" * 60)

data_reg = df[
    [
        "AP",
        "consensus_divergence",
        "log_n"
    ]
].dropna()

y = data_reg[
    "AP"
].values

X = np.column_stack(
    [
        np.ones(len(data_reg)),
        data_reg["consensus_divergence"].values,
        data_reg["log_n"].values
    ]
)


coeffs, residuals, rank, singular_values = lstsq(
    X,
    y,
    rcond=None
)

intercept, beta_div, beta_n = coeffs


# OLS goodness-of-fit and coefficient uncertainty
y_pred = X @ coeffs

ss_res = np.sum(
    (y - y_pred) ** 2
)

ss_tot = np.sum(
    (y - y.mean()) ** 2
)

r2 = 1 - (
    ss_res / ss_tot
)

n_reg = len(data_reg)
k = X.shape[1]

sigma2 = (
    ss_res /
    (n_reg - k)
)

cov_matrix = (
    sigma2 *
    inv(X.T @ X)
)

se = np.sqrt(
    np.diag(cov_matrix)
)

t_stats = (
    coeffs /
    se
)

p_vals = (
    2 *
    t_dist.sf(
        np.abs(t_stats),
        df=n_reg - k
    )
)


print(
    f"OLS: AP ~ divergence + log10(n_train)"
)

print(
    f"n = {n_reg}"
)

print(
    f"R² = {r2:.4f}"
)

print(
    f"\n{'Term':<20}"
    f"{'β':>10}"
    f"{'SE':>10}"
    f"{'t':>10}"
    f"{'p':>12}"
)

for name, beta, se_i, t_i, p_i in zip(
    [
        "intercept",
        "divergence",
        "log10(n_train)"
    ],
    coeffs,
    se,
    t_stats,
    p_vals
):

    print(
        f"{name:<20}"
        f"{beta:>10.4f}"
        f"{se_i:>10.4f}"
        f"{t_i:>10.3f}"
        f"{p_i:>12.3e}"
    )


if p_vals[1] < 0.05 and beta_div < 0:

    ols_interp = (
        "Divergence remains a significant negative predictor "
        "of AP after controlling for training coverage."
    )

elif p_vals[1] < 0.05:

    ols_interp = (
        "Divergence remains significant in the model, but the "
        "coefficient is positive."
    )

else:

    ols_interp = (
        "Divergence is not a significant independent predictor "
        "of AP after controlling for training coverage."
    )


summary_rows.append(
    {
        "analysis": "OLS: AP ~ divergence + log10(n_train)",
        "metric_x": "divergence + log10(n_train)",
        "metric_y": "AP",
        "rho_or_beta": beta_div,
        "p_value": p_vals[1],
        "n": n_reg,
        "interpretation": (
            f"{ols_interp} R²={r2:.3f}."
        )
    }
)


# Plot regression fit and coefficient estimates
fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


axes[0].scatter(
    y_pred,
    y,
    color="#2166ac",
    alpha=0.7,
    edgecolors="white",
    linewidths=0.3,
    s=50
)

lims = [
    min(
        y.min(),
        y_pred.min()
    ) - 0.02,

    max(
        y.max(),
        y_pred.max()
    ) + 0.02
]

axes[0].plot(
    lims,
    lims,
    "k--",
    linewidth=1,
    alpha=0.6,
    label="y=x"
)

axes[0].set_xlim(
    lims
)

axes[0].set_ylim(
    lims
)

axes[0].set_xlabel(
    "Predicted AP",
    fontsize=10
)

axes[0].set_ylabel(
    "Observed AP",
    fontsize=10
)

axes[0].set_title(
    f"OLS predicted vs observed\nR² = {r2:.3f}",
    fontsize=10
)

axes[0].grid(
    alpha=0.3
)

axes[0].legend(
    fontsize=9
)


terms = [
    "divergence",
    "log10(n_train)"
]

betas = [
    beta_div,
    beta_n
]

ses_terms = [
    se[1],
    se[2]
]

ps_terms = [
    p_vals[1],
    p_vals[2]
]


coefficient_colors = [
    "#d6604d"
    if p < 0.05
    else "#aaaaaa"
    for p in ps_terms
]


axes[1].barh(
    terms,
    betas,
    xerr=ses_terms,
    color=coefficient_colors,
    edgecolor="white",
    linewidth=0.5,
    capsize=4
)

axes[1].axvline(
    0,
    color="black",
    linewidth=0.8
)

axes[1].set_xlabel(
    "Regression coefficient (β)",
    fontsize=10
)

axes[1].set_title(
    "Independent predictors of AP",
    fontsize=10
)


for i, (beta, p) in enumerate(
    zip(
        betas,
        ps_terms
    )
):

    offset = (
        0.002
        if beta >= 0
        else -0.002
    )

    axes[1].text(
        beta + offset,
        i,
        f"p={p:.2e}",
        va="center",
        ha="left" if beta >= 0 else "right",
        fontsize=8
    )

axes[1].grid(
    axis="x",
    alpha=0.3
)


plt.suptitle(
    "Multiple regression: AP ~ divergence + log10(n_train)",
    fontsize=10
)

plt.tight_layout()

plt.savefig(
    OUT_F / "confound_control_regression.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved: confound_control_regression.png"
)


# Save statistical summary
summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUT_T / "confound_control_summary.csv",
    index=False
)


# Final interpretation
print("\n" + "=" * 60)
print("OVERALL INTERPRETATION")
print("=" * 60)


div_sig_partial = (
    p_partial < 0.05
    and
    r_partial < 0
)

div_sig_high_coverage = any(
    row["p"] < 0.05
    and
    row["rho"] < 0
    for row in strat_rows
    if (
        "Q3" in row["quartile"]
        or
        "Q4" in row["quartile"]
    )
)

div_sig_ols = (
    p_vals[1] < 0.05
    and
    beta_div < 0
)

coverage_sig = (
    p_n < 0.05
)


print(
    f"\nCoverage predicts AP: "
    f"{coverage_sig}"
)

print(
    f"Divergence survives partial correlation: "
    f"{div_sig_partial}"
)

print(
    f"Divergence significant in high-coverage strata: "
    f"{div_sig_high_coverage}"
)

print(
    f"Divergence significant in OLS: "
    f"{div_sig_ols}"
)


if div_sig_partial and div_sig_ols:

    verdict = (
        "STRONG: Sequence divergence remains a negative predictor "
        "of AP after controlling for training coverage. This "
        "strengthens the representation bottleneck hypothesis."
    )

elif div_sig_partial or div_sig_ols:

    verdict = (
        "MODERATE: Divergence retains some evidence of an "
        "independent association with AP after confound control, "
        "but the separation from training coverage is incomplete."
    )

else:

    verdict = (
        "NULL: The divergence-AP relationship does not survive "
        "control for training coverage. The original association "
        "may be substantially explained by data density."
    )


print(
    f"\n{verdict}"
)

print(
    f"\nSaved: {OUT_T / 'confound_control_summary.csv'}"
)
