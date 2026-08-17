#!/usr/bin/env python3

"""
Map residue-level physics onto biological pMHC PDB.

Uses OpenMM topology for correct chain/residue handling.
Writes a clean PDB containing only chains A/B/C.
Physics values are stored in the B-factor column.
"""
import argparse
import os
import pandas as pd
from openmm.app import PDBFile, Modeller
from openmm import unit

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    physics = {}
    for _, row in df.iterrows():

        key = (
            str(row["chain"]),
            int(row["residue_number"])
        )
        physics[key] = {
            "charge": row["mean_charge_e"],
            "mean_force": row["mean_force_kJ_mol_nm"],
            "max_force": row["max_force_kJ_mol_nm"],
        }

    pdb = PDBFile(args.pdb)

    # Select only biological chains
    keep_atoms = []

    for atom in pdb.topology.atoms():
        if atom.residue.chain.id in ["A", "B", "C"]:
            keep_atoms.append(atom.index)

    modeller = Modeller(pdb.topology, pdb.positions)

    # Remove solvent/ions
    modeller.deleteWater()

    topology = modeller.topology
    positions = modeller.positions

    # Create maps using OpenMM's own writer
    for metric in [
        "charge",
        "mean_force",
        "max_force"
    ]:

        outfile = os.path.join(args.output_dir, f"DQ25_{metric}_map.pdb")
        temp = outfile + ".tmp"
        with open(temp, "w") as f:
            PDBFile.writeFile(topology, positions,f)


        # Replace B-factor column
        output_lines = []
        with open(temp) as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21]
                    resnum = int(line[22:26])
                    value = 0.0
                    key = (chain, resnum)

                    if key in physics:
                        value = physics[key][metric]

                    # B-factor columns 61-66
                    line = (line[:60] + f"{value:6.2f}" + line[66:])
                output_lines.append(line)

        with open(outfile, "w") as f:
            f.writelines(output_lines)

        os.remove(temp)
        print("saved:", outfile)

if __name__ == "__main__":
    main()