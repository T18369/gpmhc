#!/usr/bin/env python3
"""
Extract peptide-MHC-II interface distance maps.

mhc_abs_pos = True MHC sequence residue position.
pseudo_pos = Graph-pMHC schema residue index.
concat_pos = Graph-pMHC alpha+beta node ordering.

Outputs:
experiment1/distance_maps/
    binary/
        peptide x MHC contact maps
    continuous/
        peptide x MHC distance landscapes

    structural_metrics.csv
    alignment_metrics.csv
    extraction_qc.csv
    extraction_failures.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.PDB import (PDBParser,is_aa,)

AF2_DIR = Path("experiment1/af2_outputs")
MANIFEST = Path("experiment1/af2_job_manifest_all.csv")
MHC_SEQ_FILE = Path("gpmhc/mhc_seq_df.csv")
OUT_DIR = Path("experiment1/distance_maps")
BINARY_DIR = OUT_DIR / "binary"
CONT_DIR = OUT_DIR / "continuous"

for d in [
    OUT_DIR,
    BINARY_DIR,
    CONT_DIR,
]:

    d.mkdir(parents=True, exist_ok=True)
OVERWRITE = True

HARD_CUTOFF = 4.0
PROXIMAL_CUTOFF = 8.0

SCHEMA = {
    "DQA1": [10,11,13,24,26,34,54,55,56,63,64,67,70,71,74,78],
    "DQB1": [11,13,15,30,32,59,62,63,69,72,76,79,80,83,84,87],
    "DPA1": [8,10,21,23,30,31,42,51,52,53,57,58,61,64,65,67,68,71,72,75],
    "DPB1": [8,10,11,12,23,25,26,27,34,44,54,57,58,64,67,68,71,74,75,78,79,82,83],
    "DRA":  [6,8,10,21,23,30,31,42,51,52,53,57,58,61,64,65,67,68,71,72,75],
    "DRB1": [8,10,12,25,27,29,36,46,56,59,60,66,69,70,73,76,77,80,81,84,85,88],
    "DRB3": [8,10,12,25,27,29,36,46,56,59,60,66,69,70,73,76,77,80,81,84,85,88],
    "DRB4": [8,10,12,25,27,29,36,46,56,59,60,66,69,70,73,76,77,80,81,84,85,88],
    "DRB5": [8,10,12,25,27,29,36,46,56,59,60,66,69,70,73,76,77,80,81,84,85,88],
}

AA3_TO_1 = {
    "ALA":"A",
    "ARG":"R",
    "ASN":"N",
    "ASP":"D",
    "CYS":"C",
    "GLN":"Q",
    "GLU":"E",
    "GLY":"G",
    "HIS":"H",
    "ILE":"I",
    "LEU":"L",
    "LYS":"K",
    "MET":"M",
    "PHE":"F",
    "PRO":"P",
    "SER":"S",
    "THR":"T",
    "TRP":"W",
    "TYR":"Y",
    "VAL":"V",
    "MSE":"M",
}

parser = PDBParser(QUIET=True)

print("Loading MHC sequences")
mhc = pd.read_csv(MHC_SEQ_FILE)
mhc = mhc[mhc["gene"].isin(SCHEMA.keys())].copy()

def normalize_allele(a):
    a = str(a)
    if "*" not in a:
        return a
    gene, rest = a.split("*", 1)
    return (gene+"*"+":".join(rest.split(":")[:2]))

mhc["allele_norm"] = (mhc["allele"].apply(normalize_allele))
SEQ_LOOKUP = dict(zip(mhc["allele_norm"],mhc["full_sequence"]))

def get_sequence(allele):
    return SEQ_LOOKUP.get(normalize_allele(allele))

def pdb_chain_sequence(chain):
    residues = [
        r
        for r in chain.get_residues()
        if is_aa(r)
    ]
    seq = "".join(
        AA3_TO_1.get(
            r.get_resname()
            .upper(),
            "X"
        )
        for r in residues
    )
    return seq, residues

def align_reference_to_pdb(
    reference,
    pdb_seq
):
    aligner = PairwiseAligner()
    aligner.mode = "local"
    alignment = aligner.align(
        reference,
        pdb_seq
    )[0]
    mapping = {}
    for ref_block, pdb_block in zip(
        alignment.aligned[0],
        alignment.aligned[1]
    ):
        for i in range(
            ref_block[1]
            -
            ref_block[0]
        ):
            mapping[ref_block[0] + i] = (pdb_block[0] + i)
    return mapping

def map_schema_positions(
    allele,
    chain
):
    gene = allele.split("*")[0]
    schema = SCHEMA[gene]
    pdb_seq, pdb_res = (pdb_chain_sequence(chain))
    ref_seq = get_sequence(allele)

    if ref_seq is None:
        raise ValueError(f"Missing sequence: {allele}")

    mapping = align_reference_to_pdb(ref_seq,pdb_seq)
    mapped = []

    for pseudo_pos, abs_pos in enumerate(
        schema,
        start=1
    ):
        ref_index = abs_pos - 1
        pdb_index = mapping.get(
            ref_index
        )
        if pdb_index is None:
            mapped.append(
                {
                    "pseudo_pos":pseudo_pos,
                    "mhc_abs_pos":abs_pos,
                    "residue":None,
                    "aa":None,
                    "mapped":False
                }
            )
        else:
            mapped.append(
                {
                    "pseudo_pos":pseudo_pos,
                    "mhc_abs_pos":abs_pos,
                    "residue":pdb_res[pdb_index],
                    "aa":pdb_seq[pdb_index],
                    "mapped":True
                }
            )
    return mapped

def identify_chains(structure):
    chains = []
    for chain in structure[0].get_chains():
        residues = [
            r
            for r in chain.get_residues()
            if is_aa(r)
        ]
        if residues:
            chains.append(
                (chain,residues)
            )

    if len(chains) != 3:
        raise ValueError(f"Expected 3 protein chains, found {len(chains)}")
    chains.sort(key=lambda x: len(x[1]))
    peptide = chains[0]
    mhc = chains[1:]
    return (
        peptide[0],
        peptide[1],
        mhc[0][0],
        mhc[1][0]
    )

def heavy_atom_distance(residue1,residue2):
    atoms1 = [
        a
        for a in residue1.get_atoms()
        if a.element
        and a.element != "H"
    ]
    atoms2 = [
        a
        for a in residue2.get_atoms()
        if a.element
        and a.element != "H"
    ]

    if (len(atoms1) == 0 or len(atoms2) == 0):
        return np.nan
    return float(
        min(
            a - b
            for a in atoms1
            for b in atoms2
        )
    )

def combined_schema(
    alpha,
    beta,
    alpha_map,
    beta_map
):
    rows = []
    alpha_gene = alpha.split("*")[0]
    beta_gene = beta.split("*")[0]
    alpha_len = len(
        SCHEMA[alpha_gene]
    )

    # alpha chain first
    for x in alpha_map:
        rows.append(
            {
                **x,
                "mhc_gene":alpha_gene,
                "mhc_chain_type":"alpha",
                "concat_pos":x["pseudo_pos"]
            }
        )

    # beta chain second
    for x in beta_map:
        rows.append(

            {

                **x,

                "mhc_gene":
                    beta_gene,

                "mhc_chain_type":
                    "beta",

                "concat_pos":
                    alpha_len + x["pseudo_pos"]
            }
        )
    return rows

def locate_pdb(job_name):
    hits = list(AF2_DIR.rglob(f"{job_name}*rank_001*.pdb"))
    if not hits:
        return None
    return hits[0]

def process_structure(job):
    job_name = str(job["job_name"])
    pdb = locate_pdb(job_name)
    if pdb is None:
        raise FileNotFoundError(job_name)
    alpha = str(job["alpha_chain"])
    beta = str(job["beta_chain"])
    structure = parser.get_structure(job_name,pdb)

    (peptide_chain,
     peptide_res,
     alpha_chain,
     beta_chain
    ) = identify_chains(structure)

    alpha_map = map_schema_positions(alpha, alpha_chain)
    beta_map = map_schema_positions(beta, beta_chain)
    schema = combined_schema(
        alpha,
        beta,
        alpha_map,
        beta_map
    )
    rows = []

    for pep_pos, pep in enumerate(
        peptide_res,
        start=1
    ):
        for mhc in schema:
            if not mhc["mapped"]:
                distance = np.nan
            else:
                distance = heavy_atom_distance(
                    pep,
                    mhc["residue"]
                )
            rows.append(
                {"job_name":job_name,
                    "peptide":job["peptide"],
                    "pep_pos":pep_pos,
                    "pep_aa":AA3_TO_1.get(pep.get_resname(),
                    "X"),

                    # Graph-pMHC coordinates
                    "pseudo_pos":mhc["pseudo_pos"],
                    "concat_pos":mhc["concat_pos"],
                    # TRUE structural coordinate
                    "mhc_abs_pos":mhc["mhc_abs_pos"],
                    "mhc_gene":mhc["mhc_gene"],
                    "mhc_chain_type":mhc["mhc_chain_type"],
                    "distance_A":distance,
                    "hard_contact":
                        bool(distance <= HARD_CUTOFF)
                        if not np.isnan(distance)
                        else False,
                    "proximal":
                        bool(HARD_CUTOFF < distance <= PROXIMAL_CUTOFF)
                        if not np.isnan(distance)
                        else False
                }
            )
    return pd.DataFrame(rows)

print("Loading manifest")
manifest = pd.read_csv(MANIFEST)
structural_metrics = []
qc_rows = []
failures = []

for i, (_, job) in enumerate(
    manifest.iterrows(),
    start=1
):
    job_name = str(job["job_name"])
    print(f"[{i}/{len(manifest)}] {job_name}")

    binary_file = (BINARY_DIR /f"{job_name[:100]}.csv")
    continuous_file = (CONT_DIR /f"{job_name[:100]}.csv")

    try:
        distances = process_structure(job)
        distances = distances[
            distances["distance_A"].notna()
        ].copy()

        continuous_cols = [
            "job_name",
            "peptide",
            "pep_pos",
            "pep_aa",
            "pseudo_pos",
            "concat_pos",
            "mhc_abs_pos",
            "mhc_gene",
            "mhc_chain_type",
            "distance_A",
            "hard_contact",
            "proximal"
        ]

        distances[continuous_cols].to_csv(
            continuous_file,
            index=False
        )

        binary = distances.copy()
        binary["contact"] = (
            binary["hard_contact"]
            .astype(int)
        )

        binary_cols = [
            "job_name",
            "peptide",
            "pep_pos",
            "pep_aa",
            "mhc_abs_pos",
            "mhc_gene",
            "contact"
        ]

        binary[binary_cols].to_csv(
            binary_file,
            index=False
        )

        structural_metrics.append(
            {
                "job_name":job_name,
                "peptide_length":distances.pep_pos.nunique(),
                "mhc_positions":distances.mhc_abs_pos.nunique(),
                "pairs":len(distances),
                "hard_contacts":int(distances.hard_contact.sum()),
                "proximal_contacts":int(distances.proximal.sum())
            }
        )

        qc_rows.append(
            {
                "job_name":job_name,
                "status":"success",
                "rows":len(distances),
                "mhc_abs_min":distances.mhc_abs_pos.min(),
                "mhc_abs_max":distances.mhc_abs_pos.max()
            }
        )

        print(
            f"  OK | "
            f"pep={distances.pep_pos.nunique()} "
            f"MHC={distances.mhc_abs_pos.min()}-"
            f"{distances.mhc_abs_pos.max()}"
        )

    except Exception as e:
        failures.append(
            {
                "job_name":job_name,
                "error":str(e)
            }
        )
        print(f" FAIL | {e}")

pd.DataFrame(structural_metrics).to_csv(OUT_DIR /"structural_metrics.csv", index=False)
pd.DataFrame(qc_rows).to_csv(OUT_DIR /"extraction_qc.csv", index=False)
pd.DataFrame(failures).to_csv(OUT_DIR /"extraction_failures.csv", index=False)

print("Extraction complete")
print(f"Successful: {len(structural_metrics)}")
print(f"Failed: {len(failures)}")
print(f"Output: {OUT_DIR}")
