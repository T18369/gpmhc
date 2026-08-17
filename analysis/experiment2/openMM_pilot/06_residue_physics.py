#!/usr/bin/env python3

"""
Aggregate atom-level OpenMM features into residue-level physicochemical features.

Input:
    *_atom_features.csv

Output:
    *_residue_physics.csv

Features:
    - residue identity
    - chain
    - component (MHC / peptide / water / ion)
    - mean partial charge
    - total partial charge
    - mean force magnitude
    - max force magnitude
    - atom count
"""
import argparse
import os
import pandas as pd

def main():

    parser = argparse.ArgumentParser(
        description="Aggregate atom physics into residue-level features."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Atom features CSV from OpenMM extraction."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(
        os.path.basename(args.input)
    )[0].replace("_atom_features", "")

    df = pd.read_csv(args.input)
    residue_features = (
        df.groupby(
            [
                "chain",
                "residue_number",
                "residue",
                "component",
            ],
            dropna=False,
        )
        .agg(
            atom_count=("atom_idx", "count"),
            total_charge_e=("partial_charge_e", "sum"),
            mean_charge_e=("partial_charge_e", "mean"),
            mean_force_kJ_mol_nm=("force_magnitude_kJ_mol_nm", "mean"),
            max_force_kJ_mol_nm=("force_magnitude_kJ_mol_nm", "max"),
        )
        .reset_index()
    )

    # Remove solvent/ions for biological analysis.
    biological = residue_features[
        residue_features["component"].isin(["mhc", "peptide",])
    ]

    output = os.path.join(args.output_dir, f"{base}_residue_physics.csv",)
    residue_features.to_csv(output, index=False,)
    biological_output = os.path.join(
        args.output_dir,
        f"{base}_biological_residue_physics.csv",
    )
    biological.to_csv(biological_output, index=False,)

    print(f"Total residues: {len(residue_features)}")
    print(f"Biological residues: {len(biological)}")
    print("Biological residue summary:")
    print(biological.groupby("component").size())
    print(f"Residue table: {output}")
    print(f"Biological table: {biological_output}")

if __name__ == "__main__":
    main()