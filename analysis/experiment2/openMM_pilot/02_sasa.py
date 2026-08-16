#!/usr/bin/env python3
import argparse
import os

import mdtraj as md
import pandas as pd
import freesasa

def main():
    parser = argparse.ArgumentParser(
        description="Calculate atom- and residue-level SASA."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    # MDTraj SASA
    traj = md.load(args.input)

    sasa = md.shrake_rupley(
        traj,
        mode="atom",
        probe_radius=0.14,
    )[0]

    atom_table = []

    for atom, value in zip(traj.topology.atoms, sasa):
        atom_table.append(
            {
                "atom_idx": atom.index,
                "atom_name": atom.name,
                "residue": atom.residue.name,
                "residue_idx": atom.residue.index,
                "residue_number": atom.residue.resSeq,
                "chain": atom.residue.chain.index,
                "sasa_nm2": value,
                "sasa_A2": value * 100.0,
            }
        )

    atom_df = pd.DataFrame(atom_table)

    # Residue-level SASA
    residue_df = (
        atom_df
        .groupby(
            ["chain", "residue_idx", "residue_number", "residue"],
            as_index=False,
        )["sasa_A2"]
        .sum()
    )

    atom_output = os.path.join(args.output_dir,f"{base}_atom_sasa.csv",)
    residue_output = os.path.join(args.output_dir,f"{base}_residue_sasa.csv",)

    atom_df.to_csv(atom_output, index=False)
    residue_df.to_csv(residue_output, index=False)

    # FreeSASA independent calculation
    structure = freesasa.Structure(args.input)
    result = freesasa.calc(structure)

    summary = os.path.join(args.output_dir,f"{base}_sasa_summary.txt",)

    with open(summary, "w") as handle:
        handle.write(f"Structure: {args.input}\n")
        handle.write(f"Total SASA (A^2): {result.totalArea():.3f}\n")

    print(f"Total SASA: {result.totalArea():.3f} A^2")
    print(f"Atom SASA: {atom_output}")
    print(f"Residue SASA: {residue_output}")

if __name__ == "__main__":
    main()