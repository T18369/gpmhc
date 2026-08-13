#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import subprocess


BASE = Path("/Users/tarun/gpmhc/netmhc2pan/netmhcIIpan_train/outputs")
GPMHC = Path("/Users/tarun/gpmhc/presentation_df_w_preds.csv")

POS = BASE / "netmhc_DQ_EL_positive_final_augmentation.csv"
NEG = BASE / "netmhc_DQ_EL_negative_final_augmentation.csv"

WORK = BASE / "cdhit_analysis"
WORK.mkdir(exist_ok=True)

IDENTITY = "0.9"
WORD_SIZE = "5"


def write_fasta(peptides, outfile, prefix):
    with open(outfile, "w") as f:
        for i, pep in enumerate(peptides):
            f.write(f">{prefix}_{i}\n{pep}\n")


print("=" * 70)
print("Loading datasets")
print("=" * 70)

gp = pd.read_csv(GPMHC, low_memory=False)
pep_col = next(c for c in gp.columns if "pep" in c.lower())
gp_peptides = gp[pep_col].dropna().astype(str).unique()

pos = pd.read_csv(POS)
neg = pd.read_csv(NEG)

pos_peptides = pos["peptide"].astype(str).unique()
neg_peptides = neg["peptide"].astype(str).unique()

print(f"gpmhc peptides: {len(gp_peptides)}")
print(f"NetMHC positive: {len(pos_peptides)}")
print(f"NetMHC negative: {len(neg_peptides)}")


gp_fasta = WORK / "gpmhc.fasta"
pos_fasta = WORK / "net_positive.fasta"
neg_fasta = WORK / "net_negative.fasta"

write_fasta(gp_peptides, gp_fasta, "gpmhc")
write_fasta(pos_peptides, pos_fasta, "pos")
write_fasta(neg_peptides, neg_fasta, "neg")


def run_cdhit(fasta, output):
    cmd = ["cd-hit", "-i", str(fasta), "-o", str(output), "-c", IDENTITY, "-n", WORD_SIZE, "-d", "0"]
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


combined_fasta = WORK / "combined.fasta"

with open(combined_fasta, "w") as out:
    for fasta in [gp_fasta, pos_fasta, neg_fasta]:
        out.write(fasta.read_text())

cluster_out = WORK / "combined_cdhit"
run_cdhit(combined_fasta, cluster_out)


clusters = []
current = None

with open(f"{cluster_out}.clstr") as f:
    for line in f:
        line = line.strip()

        if line.startswith(">Cluster"):
            current = int(line.split()[1])
            continue

        name = line.split(">")[1].split("...")[0]
        clusters.append({"cluster": current, "sequence": name})

clusters = pd.DataFrame(clusters)


def source(sequence):
    if sequence.startswith("gpmhc"):
        return "gpmhc"
    if sequence.startswith("pos"):
        return "net_positive"
    if sequence.startswith("neg"):
        return "net_negative"
    return "unknown"


clusters["source"] = clusters["sequence"].apply(source)

clusters.to_csv(WORK / "cdhit_cluster_membership.csv", index=False)

summary = (
    clusters.groupby(["cluster", "source"])
    .size()
    .reset_index(name="count")
)

summary.to_csv(WORK / "cdhit_cluster_summary.csv", index=False)


cluster_sources = clusters.groupby("cluster")["source"].unique()
mixed = cluster_sources[cluster_sources.apply(lambda x: "gpmhc" in x)]


print("\n" + "=" * 70)
print("CD-HIT SUMMARY")
print("=" * 70)

print(f"Clusters: {clusters['cluster'].nunique()}")
print(f"Clusters containing gpmhc + NetMHC: {len(mixed)}")

for src in ["net_positive", "net_negative"]:
    total = clusters["source"].eq(src).sum()
    linked = clusters[clusters["cluster"].isin(mixed.index)]["source"].eq(src).sum()
    percent = linked / total * 100 if total else 0
    print(f"{src} linked to gpmhc: {linked} / {total} ({percent:.2f}%)")

print(f"\nOutputs:\n{WORK}")
print("\nDone")
