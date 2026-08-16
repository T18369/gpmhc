#!/usr/bin/env python3
"""
Add solvent and ions to protonated pMHC-II complex.

Input:
    Protonated PDB produced by Step 1 (PDB2PQR/PROPKA)

Output:
    Explicitly solvated PDB containing:
      - pMHC complex
      - TIP3P water
      - neutralizing ions
      - additional NaCl corresponding to desired ionic strength

The protonation states established at pH 5.0 are preserved.
Adding water and ions does not itself reproduce pH 5.0.
"""

from pathlib import Path
import argparse
from openmm.app import PDBFile, ForceField, Modeller
from openmm import unit
import math
def main():

    parser = argparse.ArgumentParser(
        description="Solvate a protonated pMHC-II structure with TIP3P water and ions."
    )
    parser.add_argument("--input", required=True, help="Protonated PDB from Step 1.")
    parser.add_argument("--output", required=True, help="Output solvated PDB.")
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Water-box padding in nm. Default: 1.0 nm."
    )
    parser.add_argument(
        "--ionic-strength",
        type=float,
        default=0.15,
        help="Target NaCl ionic strength in mol/L. Default: 0.15 M."
    )
    args = parser.parse_args()
    input_pdb = Path(args.input)
    output_pdb = Path(args.output)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_pdb}")
    print(f"Output: {output_pdb}")
    print(f"Water model: TIP3P")
    print(f"Box padding: {args.padding:.2f} nm")
    print(f"NaCl ionic strength: {args.ionic_strength:.2f} M")

    # Load the protonated pMHC structure.
    pdb = PDBFile(str(input_pdb))

    # TIP3P is provided by the AMBER14 OpenMM force-field distribution.
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml",)

    modeller = Modeller(pdb.topology, pdb.positions)
    initial_atoms = modeller.topology.getNumAtoms()

    print(f"Initial atoms: {initial_atoms}")

    # Add a rectangular periodic water box.
    # Padding applied around the solute so complex is surrounded by at min 
    # the requested amount of water.
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=args.padding * unit.nanometer,
        ionicStrength=args.ionic_strength * unit.molar,
        neutralize=True,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    final_atoms = modeller.topology.getNumAtoms()

    added_atoms = final_atoms - initial_atoms

    print(f"Final atoms:   {final_atoms}")
    print(f"Added atoms:   {added_atoms}")

    # Report the periodic box dimensions.
    box_vectors = modeller.topology.getPeriodicBoxVectors()

    if box_vectors is not None:
        lengths = [
            math.sqrt(
                vector.x ** 2
                + vector.y ** 2
                + vector.z ** 2
            )
            for vector in box_vectors
        ]

        print(
            "Box dimensions: "
            f"{lengths[0]:.2f} x "
            f"{lengths[1]:.2f} x "
            f"{lengths[2]:.2f} nm"
        )
    print("Writing solvated structure...")

    with open(output_pdb, "w") as handle:
        PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            handle,
            keepIds=True,
        )
    print("Solvation complete.")
if __name__ == "__main__":
    main()