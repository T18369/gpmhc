#!/usr/bin/env python3

from pathlib import Path
import pandas as pd


BASE = Path("/Users/tarun/gpmhc/netmhc2pan/netmhcIIpan_train/outputs")
POS_FILE = BASE / "netmhc_DQ_EL_positive_augmentation_clean.csv"
NEG_FILE = BASE / "netmhc_DQ_EL_negative_augmentation_clean.csv"
SEED = 42


print("Loading augmentation datasets")

pos = pd.read_csv(POS_FILE)
neg = pd.read_csv(NEG_FILE)

print(f"Positive: {len(pos)}")
print(f"Negative: {len(neg)}")


target_negative = len(pos)
core_fraction = 0.25
n_core = int(target_negative * core_fraction)

core_neg = neg[neg["similarity_class"] == "core_overlap"]
novel_neg = neg[neg["similarity_class"] == "novel"]

n_core = min(n_core, len(core_neg))
n_novel = target_negative - n_core

if n_novel > len(novel_neg):
    raise ValueError(
        f"Not enough novel negatives to balance the augmentation: "
        f"need {n_novel}, found {len(novel_neg)}"
    )

print(f"\nTarget negatives: {target_negative}")
print(f"Available core negatives: {len(core_neg)}")
print(f"Available novel negatives: {len(novel_neg)}")
print(f"Sampling {n_core} core-overlap and {n_novel} novel negatives")


core_sample = core_neg.sample(n=n_core, random_state=SEED)
novel_sample = novel_neg.sample(n=n_novel, random_state=SEED)

neg_final = (
    pd.concat([core_sample, novel_sample], ignore_index=True)
    .sample(frac=1, random_state=SEED)
    .reset_index(drop=True)
)


pos_out = BASE / "netmhc_DQ_EL_positive_final_augmentation.csv"
neg_out = BASE / "netmhc_DQ_EL_negative_final_augmentation.csv"
summary_out = BASE / "06_final_augmentation_summary.csv"

pos.to_csv(pos_out, index=False)
neg_final.to_csv(neg_out, index=False)


summary = pd.DataFrame({
    "dataset": ["positive", "negative"],
    "rows": [len(pos), len(neg_final)],
    "novel": [
        (pos["similarity_class"] == "novel").sum(),
        (neg_final["similarity_class"] == "novel").sum()
    ],
    "core_overlap": [
        (pos["similarity_class"] == "core_overlap").sum(),
        (neg_final["similarity_class"] == "core_overlap").sum()
    ]
})

print("\nFinal augmentation summary")
print(summary)

summary.to_csv(summary_out, index=False)

print("\nSaved:")
print(pos_out)
print(neg_out)
print(summary_out)

print("\nDone")
