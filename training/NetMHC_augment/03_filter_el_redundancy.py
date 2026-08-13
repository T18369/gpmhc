#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import re


REPO = Path("/Users/tarun/gpmhc")
BASE = REPO / "netmhc2pan" / "netmhcIIpan_train" / "outputs"

NET_POS = BASE / "netmhc_DQ_heterodimer_EL_positive.csv"
NET_NEG = BASE / "netmhc_DQ_heterodimer_EL_negative.csv"
GPMHC_FILE = REPO / "presentation_df_w_preds.csv"
OUT = BASE


def normalize_hla(x):
    x = str(x).upper().replace("*", "").replace(":", "").replace("-", "").replace("_", "").replace(" ", "")
    x = x.replace("DQA1", "DQA").replace("DQB1", "DQB")
    return x if x.startswith("HLA") else "HLA" + x


def extract_gpmhc_dq(x):
    alleles = re.findall(r"(DQA1\*\d+:\d+|DQB1\*\d+:\d+)", str(x))
    dqa = [a for a in alleles if "DQA" in a]
    dqb = [a for a in alleles if "DQB" in a]

    if len(dqa) == 1 and len(dqb) == 1:
        return normalize_hla(dqa[0] + dqb[0])
    return None


print("=" * 60)
print("Loading NetMHCIIpan EL")
print("=" * 60)

net_pos = pd.read_csv(NET_POS)
net_neg = pd.read_csv(NET_NEG)

net_pos["heterodimer"] = net_pos["heterodimer"].apply(normalize_hla)
net_neg["heterodimer"] = net_neg["heterodimer"].apply(normalize_hla)

print(f"Net positive: {len(net_pos)}")
print(f"Net negative: {len(net_neg)}")
print(f"Net heterodimers: {net_pos['heterodimer'].nunique()}")


print("\nLoading gpmhc")

gp = pd.read_csv(GPMHC_FILE, low_memory=False)
gp["heterodimer"] = gp["allotype"].apply(extract_gpmhc_dq)
gp = gp[gp["heterodimer"].notna()].copy()

print(f"gpmhc rows: {len(gp)}")
print(f"gpmhc DQ heterodimer rows: {len(gp)}")
print(f"gpmhc DQ heterodimers: {gp['heterodimer'].nunique()}")

print("\nNet heterodimers:")
print(sorted(net_pos["heterodimer"].unique()))

print("\ngpmhc heterodimers:")
print(sorted(gp["heterodimer"].unique()))


def compare_dataset(net, gp, label):
    print(f"\nComparing {label}")

    rows = []
    shared_alleles = set(net["heterodimer"]) & set(gp["heterodimer"])
    print(f"Shared heterodimers: {len(shared_alleles)}")

    for allele in sorted(shared_alleles):
        net_pep = set(net.loc[net["heterodimer"] == allele, "peptide"])
        gp_pep = set(gp.loc[gp["heterodimer"] == allele, "peptide"])
        shared = net_pep & gp_pep

        rows.append({
            "heterodimer": allele,
            "net_unique": len(net_pep),
            "gpmhc_unique": len(gp_pep),
            "shared": len(shared),
            "net_only": len(net_pep - gp_pep),
            "fraction_shared": len(shared) / len(net_pep) if net_pep else 0
        })

    return pd.DataFrame(rows)


pos_overlap = compare_dataset(net_pos, gp, "EL positives")
neg_overlap = compare_dataset(net_neg, gp, "EL negatives")


pos_out = OUT / "EL_positive_vs_gpmhc_overlap.csv"
neg_out = OUT / "EL_negative_vs_gpmhc_overlap.csv"
summary_out = OUT / "EL_gpmhc_redundancy_summary.csv"

pos_overlap.to_csv(pos_out, index=False)
neg_overlap.to_csv(neg_out, index=False)

summary = pd.DataFrame({
    "dataset": ["EL_positive", "EL_negative"],
    "heterodimers_compared": [len(pos_overlap), len(neg_overlap)],
    "mean_fraction_shared": [
        pos_overlap["fraction_shared"].mean() if len(pos_overlap) else 0,
        neg_overlap["fraction_shared"].mean() if len(neg_overlap) else 0
    ]
})

summary.to_csv(summary_out, index=False)


print("\nSummary")
print(summary)

if len(pos_overlap):
    print("\nTop positive overlap")
    print(pos_overlap.sort_values("fraction_shared", ascending=False).head(20))

if len(neg_overlap):
    print("\nTop negative overlap")
    print(neg_overlap.sort_values("fraction_shared", ascending=False).head(20))

print("\nDone")
