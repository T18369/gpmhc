#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

REPO = Path("/Users/tarun/gpmhc")
AUG_FILE = REPO / "HLAII_NetMHC_DQ_added_only.csv"

print("Loading augmentation")

df = pd.read_csv(AUG_FILE)
df["peptide"] = df["peptide"].astype(str)
df["length"] = df["peptide"].str.len()

print(f"Total rows:       {len(df):,}")
print(f"Unique peptides:  {df['peptide'].nunique():,}")
print(f"Unique allotypes: {df['allotype'].nunique():,}")

print("\nLength distribution")

for length, count in df["length"].value_counts().sort_index().items():
    print(f"{length:2d} aa : {count:6,d}")

print("\nCandidate peptide counts")

for cutoff in [11, 15, 19]:
    print(f">= {cutoff:2d} aa : {(df['length'] >= cutoff).sum():6,d}")


print("\nPotential 9-mer windows")

for cutoff in [11, 15, 19]:
    subset = df[df["length"] >= cutoff]
    total_windows = (subset["length"] - 8).sum()

    print(
        f">= {cutoff:2d} aa : {len(subset):6,d} peptides  ->  "
        f"{int(total_windows):8,d} possible 9-mer windows"
    )


print("\nLength statistics")
print(df["length"].describe())
print("\nDone.")
