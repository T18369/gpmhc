#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/tarun/gpmhc")
TRAIN_FILE = REPO / "HLAII_train.csv"
AUG_FILE = (
    REPO / "netmhc2pan" / "NetMHCIIpan_train" / "outputs" /
    "dq_sliding_window_augmentation" / "HLAII_DQ_window_augmented_only.csv"
)
OUT_FILE = REPO / "HLAII_train_DQwindow_augmented.csv"

print("Loading HLAII heterodimer training set")

train = pd.read_csv(TRAIN_FILE, low_memory=False)
SCHEMA = train.columns.tolist()

print(f"Original train rows: {len(train)}")
print(f"Original columns: {len(SCHEMA)}")

if "Unnamed: 0" not in SCHEMA or "Unnamed: 0.1" not in SCHEMA:
    raise RuntimeError("Required ID columns are missing")

train["Unnamed: 0"] = pd.to_numeric(train["Unnamed: 0"], errors="coerce")
max_id = int(train["Unnamed: 0"].max())

print(f"\nExisting ID range: {train['Unnamed: 0'].min()} - {max_id}")
print("\nLoading DQ augmentation")
if not AUG_FILE.exists():
    raise FileNotFoundError(f"Missing:\n{AUG_FILE}")

aug = pd.read_csv(AUG_FILE, low_memory=False)
print(f"DQ augmentation rows: {len(aug)}")

# Drop accidental pandas index columns before schema matching.
extra_index = [c for c in aug.columns if c.startswith("Unnamed: 0.1")]
if extra_index:
    print(f"Removing accidental columns: {extra_index}")
    aug = aug.drop(columns=extra_index)


print("\nAligning augmentation schema")
for col in SCHEMA:
    if col not in aug.columns:
        aug[col] = np.nan

extra = [c for c in aug.columns if c not in SCHEMA]
if extra:
    print(f"Removing extra columns: {extra}")
    aug = aug.drop(columns=extra)

aug = aug[SCHEMA]

if not aug.columns.equals(train.columns):
    raise RuntimeError("Augmentation schema does not match HLAII_train")

print("\nAssigning appended IDs")
aug["Unnamed: 0.1"] = np.nan
aug["Unnamed: 0"] = np.arange(max_id + 1, max_id + 1 + len(aug), dtype=int)
aug["split"] = "train"

print(f"New ID range: {aug['Unnamed: 0'].min()} - {aug['Unnamed: 0'].max()}")
print("\nAppending")
train_aug = pd.concat([train, aug], ignore_index=True)[SCHEMA]
print("\nValidation")

duplicate_ids = train_aug["Unnamed: 0"].duplicated().sum()
populated_index = train_aug["Unnamed: 0.1"].notna().sum()

print(f"Final rows: {len(train_aug)}")
print(f"Final columns: {len(train_aug.columns)}")
print(f"Schema preserved: {train_aug.columns.equals(train.columns)}")
print(f"Unnamed: 0.1 populated: {populated_index}")
print(f"Duplicate IDs: {duplicate_ids}")

if duplicate_ids:
    raise RuntimeError("Duplicate IDs detected")

if populated_index:
    raise RuntimeError("Unnamed: 0.1 contains values")

train_aug.to_csv(OUT_FILE, index=False)
print(f"\nSaved:\n{OUT_FILE}")
print("\nDone.")
