#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import pandas as pd


BASE = Path("/Users/tarun/gpmhc")
EL_DIR = BASE / "netmhc2pan" / "netmhcIIpan_train" / "outputs"
GPMHC_FILE = BASE / "presentation_df_w_preds.csv"
OUT = EL_DIR

K_VALUES = [9, 8]


def normalize_dq(x):
    x = str(x).upper().replace("*", "").replace(":", "").replace("-", "").replace("_", "")
    return x if x.startswith("HLA") else "HLA" + x if "DQA" in x and "DQB" in x else None


def kmers(peptide, k):
    peptide = str(peptide)
    return {peptide[i:i + k] for i in range(len(peptide) - k + 1)} if len(peptide) >= k else set()


def extract_dq(allotype):
    parts = str(allotype).replace("___", "_").split("_")
    dqa = next((p for p in parts if p.startswith("DQA1")), None)
    dqb = next((p for p in parts if p.startswith("DQB1")), None)
    return normalize_dq(dqa + dqb) if dqa and dqb else None


print("=" * 70)
print("Loading gpmhc")
print("=" * 70)

gp = pd.read_csv(GPMHC_FILE, low_memory=False)
gp = gp[gp["allotype"].astype(str).str.contains("DQA", na=False)].copy()
gp["heterodimer"] = gp["allotype"].apply(extract_dq)
gp = gp[gp["heterodimer"].notna()].copy()

print(f"gpmhc DQ rows: {len(gp)}")
print(f"gpmhc heterodimers: {gp['heterodimer'].nunique()}")


print("\nBuilding gpmhc indices")

exact_index = defaultdict(set)
kmer_index = {k: defaultdict(set) for k in K_VALUES}

for row in gp.itertuples(index=False):
    hla = row.heterodimer
    pep = str(row.peptide)

    exact_index[hla].add(pep)

    for k in K_VALUES:
        kmer_index[k][hla].update(kmers(pep, k))

print(f"Indexed heterodimers: {len(exact_index)}")


def screen(el_file, label):
    print(f"\n{'=' * 70}\nProcessing {label}\n{'=' * 70}")

    df = pd.read_csv(el_file)
    df["heterodimer"] = df["heterodimer"].apply(normalize_dq)

    print(f"Rows: {len(df)}")
    print(f"Net heterodimers: {df['heterodimer'].nunique()}")

    results = []

    for hla, group in df.groupby("heterodimer"):
        if hla not in exact_index:
            continue

        gp_exact = exact_index[hla]
        gp_kmers = {k: kmer_index[k][hla] for k in K_VALUES}

        for pep in group["peptide"].astype(str):
            results.append({
                "heterodimer": hla,
                "peptide": pep,
                "exact_match": pep in gp_exact,
                "k9_match": bool(kmers(pep, 9) & gp_kmers[9]),
                "k8_match": bool(kmers(pep, 8) & gp_kmers[8])
            })

    out = pd.DataFrame(results)
    outfile = OUT / f"kmer_similarity_{label}.csv"
    out.to_csv(outfile, index=False)

    print(f"Compared peptides: {len(out)}")

    if not out.empty:
        print(f"Exact matches: {out['exact_match'].sum()}")
        print(f"9mer matches: {out['k9_match'].sum()}")
        print(f"8mer matches: {out['k8_match'].sum()}")

    print(f"Saved: {outfile}")
    return out


positive = screen(
    EL_DIR / "netmhc_DQ_EL_positive_exact_clean.csv",
    "positive"
)

negative = screen(
    EL_DIR / "netmhc_DQ_EL_negative_exact_clean.csv",
    "negative"
)

print("\nDone")
