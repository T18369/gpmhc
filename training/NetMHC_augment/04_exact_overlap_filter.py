#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


BASE = Path("/Users/tarun/gpmhc/netmhc2pan/netmhcIIpan_train")
OUT = BASE / "outputs"

NET_POS = OUT / "netmhc_DQ_heterodimer_EL_positive.csv"
NET_NEG = OUT / "netmhc_DQ_heterodimer_EL_negative.csv"
GPMHC = Path("/Users/tarun/gpmhc/presentation_df_w_preds.csv")


def normalize_hla(x):
    return str(x).upper().replace("*", "").replace(":", "").replace("-", "").replace("_", "")


def normalize_peptide(x):
    return str(x).upper().strip()


print("=" * 60)
print("Loading NetMHCIIpan EL")
print("=" * 60)

pos = pd.read_csv(NET_POS)
neg = pd.read_csv(NET_NEG)

print(f"Positive: {len(pos)}")
print(f"Negative: {len(neg)}")

for df in [pos, neg]:
    df["peptide"] = df["peptide"].apply(normalize_peptide)
    df["heterodimer"] = df["heterodimer"].apply(normalize_hla)


print("\nLoading gpmhc")

gp = pd.read_csv(GPMHC, low_memory=False)
print(f"gpmhc rows: {len(gp)}")

gp = gp[gp["allotype"].astype(str).str.contains("DQA1", na=False)].copy()
gp["heterodimer"] = gp["allotype"].apply(normalize_hla)
gp["peptide"] = gp["peptide"].apply(normalize_peptide)

gp_pairs = set(zip(gp["heterodimer"], gp["peptide"]))
print(f"gpmhc DQ peptide pairs: {len(gp_pairs)}")


def remove_exact_overlap(df, name):
    print(f"\nProcessing {name}")

    before = len(df)
    pairs = zip(df["heterodimer"], df["peptide"])
    clean = df[[pair not in gp_pairs for pair in pairs]].copy()
    removed = before - len(clean)

    print(f"Original: {before}")
    print(f"Removed: {removed}")
    print(f"Remaining: {len(clean)}")
    print(f"Removal %: {removed / before * 100:.3f}")

    return clean


pos_clean = remove_exact_overlap(pos, "EL positive")
neg_clean = remove_exact_overlap(neg, "EL negative")


pos_out = OUT / "netmhc_DQ_EL_positive_exact_clean.csv"
neg_out = OUT / "netmhc_DQ_EL_negative_exact_clean.csv"

pos_clean.to_csv(pos_out, index=False)
neg_clean.to_csv(neg_out, index=False)

print("\nSaved:")
print(pos_out)
print(neg_out)

print("\nFinal summary")
print("=" * 60)
print(f"Positive examples: {len(pos_clean)}")
print(f"Negative examples: {len(neg_clean)}")
print("\nReady for sequence similarity analysis: k-mer clustering / CD-HIT")
