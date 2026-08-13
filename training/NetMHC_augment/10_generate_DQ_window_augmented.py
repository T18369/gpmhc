#!/usr/bin/env python3

from pathlib import Path
import re
import numpy as np
import pandas as pd
REPO = Path("/Users/tarun/gpmhc")
TRAIN_FILE = REPO / "HLAII_train.csv"
OUT_DIR = REPO / "netmhc2pan" / "NetMHCIIpan_train" / "outputs"
POS_FILE = OUT_DIR / "netmhc_DQ_EL_positive_final_augmentation.csv"
NEG_FILE = OUT_DIR / "netmhc_DQ_EL_negative_final_augmentation.csv"
OUT_FILE = OUT_DIR / "DQ_window_final" / "HLAII_train_DQwindow_augmented.csv"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading HLAII_train")
train = pd.read_csv(TRAIN_FILE, low_memory=False)
SCHEMA = train.columns.tolist()
print(f"Rows: {len(train)}")
print(f"Columns: {len(SCHEMA)}")
print(f"\nSchema:\n{SCHEMA}")

required = ["Unnamed: 0", "Unnamed: 0.1", "peptide", "nFlank", "cFlank", "allotype", "split"]
missing = [col for col in required if col not in SCHEMA]

if missing:
    raise RuntimeError(f"Missing columns: {missing}")

train["Unnamed: 0"] = pd.to_numeric(train["Unnamed: 0"], errors="coerce")
max_id = int(train["Unnamed: 0"].max())

print(f"\nStarting ID: {max_id}")
def normalize_allotype(x):
    match = re.search(r"DQA1(\d{2})(\d{2})DQB1(\d{2})(\d{2})", str(x))
    if match:
        return (
            f"DQA1*{match.group(1)}:{match.group(2)}___"
            f"DQB1*{match.group(3)}:{match.group(4)}"
        )
    return str(x)

def generate_windows(df):
    rows = []

    for row in df.itertuples(index=False):
        peptide = str(row.peptide)
        allotype = normalize_allotype(row.heterodimer)

        if len(peptide) < 19:
            continue

        for i in range(len(peptide) - 8):
            n_flank = peptide[i - 5:i] if i >= 5 else None
            c_flank = peptide[i + 9:i + 14] if i + 14 <= len(peptide) else None

            if n_flank is None or c_flank is None:
                continue

            if len(n_flank) == 5 and len(c_flank) == 5:
                rows.append({
                    "peptide": peptide[i:i + 9],
                    "nFlank": n_flank,
                    "cFlank": c_flank,
                    "allotype": allotype,
                    "EL": int(row.EL)
                })

    return pd.DataFrame(rows)


print("\nLoading raw NetMHC")
pos = pd.read_csv(POS_FILE)
neg = pd.read_csv(NEG_FILE)
raw = pd.concat([pos, neg], ignore_index=True)
print(f"Raw rows: {len(raw)}")
print("\nGenerating 9-mer windows")

windows = generate_windows(raw)

if windows.empty:
    raise RuntimeError("No valid windows generated")

print(f"Generated windows: {len(windows)}")

windows = windows.drop_duplicates().reset_index(drop=True)
print(f"After duplicate removal: {len(windows)}")


print("\nFormatting augmentation")
aug = pd.DataFrame(np.nan, index=range(len(windows)), columns=SCHEMA)

aug["peptide"] = windows["peptide"]
aug["nFlank"] = windows["nFlank"]
aug["cFlank"] = windows["cFlank"]
aug["allotype"] = windows["allotype"]
aug["molecule"] = "DQA1,DQB1"
aug["data_type"] = "EL"
aug["EL"] = windows["EL"]
aug["split"] = "train"
aug["is_multiallotypic"] = False
aug["peptide_length"] = 9
aug["concat"] = aug["nFlank"] + aug["peptide"] + aug["cFlank"]
aug["cluster_label"] = -1
aug["analysis_id"] = -1
aug["Unnamed: 0.1"] = np.nan
aug["Unnamed: 0"] = np.arange(max_id + 1, max_id + 1 + len(aug), dtype=int)

print("\nAppending")
final = pd.concat([train, aug], ignore_index=True)[SCHEMA]
appended = final.iloc[len(train):]

print("\nValidation")
print("-" * 70)
print(f"Final rows: {len(final)}")
print(f"Added: {len(aug)}")
print(f"Schema: {final.columns.equals(train.columns)}")
print(f"Original Unnamed: 0.1 populated: {train['Unnamed: 0.1'].notna().sum()}")
print(f"Appended Unnamed: 0.1 populated: {appended['Unnamed: 0.1'].notna().sum()}")
print(f"ID dtype: {final['Unnamed: 0'].dtype}")
print(f"Appended ID range: {appended['Unnamed: 0'].min()} - {appended['Unnamed: 0'].max()}")
print(f"Duplicate appended IDs: {appended['Unnamed: 0'].duplicated().sum()}")

id_overlap = len(set(appended["Unnamed: 0"]) & set(train["Unnamed: 0"]))
print(f"ID overlap: {id_overlap}")

if not final.columns.equals(train.columns):
    raise RuntimeError("Schema mismatch")
if appended["Unnamed: 0.1"].notna().any():
    raise RuntimeError("Appended rows contain values in Unnamed: 0.1")
if appended["Unnamed: 0"].duplicated().any():
    raise RuntimeError("Duplicate appended IDs")
if id_overlap:
    raise RuntimeError("Appended IDs overlap existing IDs")

final.to_csv(OUT_FILE, index=False)

print(f"\nSaved:\n{OUT_FILE}")
print("\nDone.")
