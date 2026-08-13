#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


BASE = Path("/Users/tarun/gpmhc/netmhc2pan/netmhcIIpan_train")
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

MAP_FILE = OUT / "dq_heterodimer_sample_map.csv"
EL_FILES = sorted(BASE.glob("*_el"))


print("Loading heterodimer map")
mapping = pd.read_csv(MAP_FILE)
sample_to_heterodimer = dict(zip(mapping["sample"], mapping["heterodimer"]))
print(f"Mapped samples: {len(sample_to_heterodimer)}")


print("\nExtracting EL dataset")
records = []

for f in EL_FILES:
    print(f"  {f.name}")

    df = pd.read_csv(
        f,
        sep=r"\s+",
        header=None,
        names=["peptide", "EL", "sample", "context"]
    )

    df = df[df["sample"].isin(sample_to_heterodimer)]
    if df.empty:
        continue

    df["heterodimer"] = df["sample"].map(sample_to_heterodimer)
    records.append(df[["peptide", "EL", "sample", "heterodimer"]])

EL = pd.concat(records, ignore_index=True)


print("\nEL dataset")
print(f"Raw extracted rows: {len(EL)}")
print(f"EL positives: {(EL['EL'] == 1).sum()}")
print(f"EL negatives: {(EL['EL'] == 0).sum()}")
print(f"Unique samples: {EL['sample'].nunique()}")
print(f"Unique heterodimers: {EL['heterodimer'].nunique()}")


before = len(EL)
EL = EL.drop_duplicates(["peptide", "heterodimer", "EL"])
after = len(EL)

print("\nDuplicate removal")
print(f"Unique peptide/heterodimer/EL pairs: {after}")
print(f"Removed duplicates: {before - after}")
print(f"Duplicate fraction: {(before - after) / before * 100:.3f}%")


EL_pos = EL[EL["EL"] == 1].copy()
EL_neg = EL[EL["EL"] == 0].copy()

print("\nFinal datasets")
print(f"Positive examples: {len(EL_pos)}")
print(f"Negative examples: {len(EL_neg)}")


coverage_cols = {
    "rows": ("peptide", "size"),
    "unique_peptides": ("peptide", "nunique"),
    "samples": ("sample", "nunique")
}

print("\nPositive heterodimer coverage")
print(EL_pos.groupby("heterodimer").agg(**coverage_cols))

print("\nNegative heterodimer coverage")
print(EL_neg.groupby("heterodimer").agg(**coverage_cols))


pos_out = OUT / "netmhc_DQ_heterodimer_EL_positive.csv"
neg_out = OUT / "netmhc_DQ_heterodimer_EL_negative.csv"

EL_pos.to_csv(pos_out, index=False)
EL_neg.to_csv(neg_out, index=False)

print("\nSaved:")
print(pos_out)
print(neg_out)
