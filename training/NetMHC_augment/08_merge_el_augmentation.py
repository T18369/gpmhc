#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd

REPO = Path("/Users/tarun/gpmhc")
DATA = REPO / "Presentation_df_w_preds.csv"
OUT_DIR = REPO / "netmhc2pan" / "netmhcIIpan_train" / "outputs"

POS_FILE = OUT_DIR / "netmhc_DQ_EL_positive_final_augmentation.csv"
NEG_FILE = OUT_DIR / "netmhc_DQ_EL_negative_final_augmentation.csv"

TRAIN_OUT = REPO / "HLAII_train_augmented.csv"
TEST_OUT = REPO / "HLAII_test_augmented.csv"
AUG_ONLY_OUT = REPO / "HLAII_NetMHC_DQ_added_only.csv"

PREDICTION_COLS = [
    "Graph-pMHC_score",
    "NetMHCIIPan-4.0",
    "MixMHCIIPred-1.2_rank",
    "MHCNuggets_rank",
]
DQ_COLS = ["mhc_dq1_1", "mhc_dq1_2", "mhc_dq1_3", "mhc_dq1_4"]

def normalize_netmhc_allotype(x):
    x = str(x).replace("HLA", "")
    match = re.search(r"DQA1(\d{2})(\d{2})DQB1(\d{2})(\d{2})", x)
    if match:
        return (
            f"DQA1*{match.group(1)}:{match.group(2)}___"
            f"DQB1*{match.group(3)}:{match.group(4)}"
        )
    return x

print("Loading Presentation_df_w_preds")
df = pd.read_csv(DATA, low_memory=False)
ORIGINAL_COLUMNS = df.columns.tolist()

print(f"Rows: {len(df)}")
print(df["split"].value_counts())

print("\nBuilding DQ lookup")

dq_source = df[["allotype", *DQ_COLS]].drop_duplicates()
dq_source = dq_source[dq_source["mhc_dq1_1"].notna()]
dq_lookup = {
    row["allotype"]: {c: row[c] for c in DQ_COLS}
    for _, row in dq_source.iterrows()
}
print(f"DQ allotypes: {len(dq_lookup)}")

print("\nLoading NetMHC EL files")
pos = pd.read_csv(POS_FILE)
neg = pd.read_csv(NEG_FILE)
print(f"Positive: {len(pos)}")
print(f"Negative: {len(neg)}")

def convert_aug(data, label):
    out = pd.DataFrame(np.nan, index=range(len(data)), columns=ORIGINAL_COLUMNS)
    out["peptide"] = data["peptide"].astype(str)
    out["allotype"] = data["heterodimer"].apply(normalize_netmhc_allotype)
    out["molecule"] = "DQA1,DQB1"
    out["data_type"] = "EL"
    out["EL"] = label
    out["split"] = "train"
    out["is_multiallotypic"] = False
    out["peptide_length"] = out["peptide"].str.len()
    out["nFlank"] = ""
    out["cFlank"] = ""
    out["concat"] = out["peptide"]
    out["cluster_label"] = -1
    out["analysis_id"] = -1

    if "mm2p_core" in data.columns:
        out["mm2p_core"] = data["mm2p_core"]
        out["peptide_core"] = data["mm2p_core"]
    else:
        out["mm2p_core"] = ""
        out["peptide_core"] = ""
    return out

aug = pd.concat([convert_aug(pos, 1), convert_aug(neg, 0)], ignore_index=True)
print(f"\nRaw augmentation: {len(aug)}")

print("\nAdding DQ pseudosequences")
for col in DQ_COLS:
    aug[col] = ""
missing = []
for idx, row in aug.iterrows():
    allo = row["allotype"]
    if allo in dq_lookup:
        for col in DQ_COLS:
            aug.loc[idx, col] = dq_lookup[allo][col]
    else:
        missing.append(allo)

print(f"Missing DQ pseudosequence rows: {len(missing)}")

if missing:
    print(f"Examples: {list(set(missing))[:20]}")

aug = aug[aug["mhc_dq1_1"] != ""].copy()
print(f"After DQ filtering: {len(aug)}")

print("\nRemoving prediction leakage")
for col in PREDICTION_COLS:
    if col in aug.columns:
        aug[col] = np.nan

aug = aug[ORIGINAL_COLUMNS]
print(f"Schema check: {aug.columns.equals(df.columns)}")

print("\nChecking exact overlap")
existing = set(zip(df["peptide"], df["allotype"], df["EL"]))
pairs = zip(aug["peptide"], aug["allotype"], aug["EL"])
keep = [(p, a, e) not in existing for p, a, e in pairs]
print(f"Removed duplicates: {len(aug) - sum(keep)}")

aug = aug[keep].copy()
print(f"Final augmentation: {len(aug)}")

if "source" not in df.columns:
    df["source"] = "Graph-pMHC_original"
if "augmentation" not in df.columns:
    df["augmentation"] = False

aug["source"] = "NetMHCIIpan_EL"
aug["augmentation"] = True

print("\nSaving augmentation-only file")
aug.to_csv(AUG_ONLY_OUT, index=False)
print(f"Saved: {AUG_ONLY_OUT}")

print("\nCreating train/test outputs")
train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

train_aug = pd.concat([train, aug], ignore_index=True)
train_aug = train_aug[ORIGINAL_COLUMNS]
test = test[ORIGINAL_COLUMNS]

print("\nFinal schema check")
print(train_aug.columns.equals(df.columns))

print("\nNetMHC rows:")
print(
    train_aug[train_aug["source"] == "NetMHCIIpan_EL"][
        ["peptide", "allotype", "EL", "concat", "cluster_label", "peptide_core"]
    ].head()
)

print("\nFinal counts")
print(f"Train: {len(train_aug)}")
print(f"Test: {len(test)}")

train_aug.to_csv(TRAIN_OUT, index=False)
test.to_csv(TEST_OUT, index=False)
print("\nSaved:")
print(TRAIN_OUT)
print(TEST_OUT)
print("\nDone")
