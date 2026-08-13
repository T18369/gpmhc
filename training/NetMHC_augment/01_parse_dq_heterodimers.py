#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

BASE = Path("/Users/tarun/gpmhc/netmhc2pan/netmhcIIpan_train")
ALLELE_FILE = BASE / "allelelist.txt"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

def normalize_dq(x):
    return (
        str(x).upper()
        .replace("*", "")
        .replace(":", "")
        .replace("-", "")
        .replace("_", "")
    )

def is_single_dq_heterodimer(x):
    x = x.upper()
    return (
        "DQA1" in x
        and "DQB1" in x
        and "DPA1" not in x
        and "DPB1" not in x
        and "DRA" not in x
        and "DRB" not in x
    )


print("Parsing allelelist.txt")

records = []

with open(ALLELE_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        sample = parts[0]
        allele_string = " ".join(parts[1:])

        if not is_single_dq_heterodimer(allele_string):
            continue

        records.append({
            "sample": sample,
            "heterodimer": normalize_dq(allele_string)
        })

df = pd.DataFrame(records)

print(f"Single DQ heterodimer samples: {len(df)}")
print(f"Unique heterodimers: {df['heterodimer'].nunique()}")
print(df.head())

outfile = OUT / "dq_heterodimer_sample_map.csv"
df.to_csv(outfile, index=False)

print(f"\nSaved: {outfile}")
