#!/usr/bin/env python3

"""
OpenMM minimized structure -> PDB2PQR -> APBS electrostatic potential

Pipeline:
1. Remove waters/ions
2. Remove non-protein chains
3. Remove hydrogens
4. Protonate with PDB2PQR at pH 5
5. Run APBS PB solver
"""
import argparse
import subprocess
from pathlib import Path
from openmm.app import (PDBFile, Modeller)

PROTEIN_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS",
    "GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE",
    "PRO","SER","THR","TRP",
    "TYR","VAL"
}

def run(cmd):
    print("\nRunning:")
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def clean_structure(pdb_file, output_file):
    print("Loading OpenMM structure...")
    pdb = PDBFile(str(pdb_file))
    modeller = Modeller(pdb.topology, pdb.positions)

    print("Removing solvent/ions...")
    modeller.deleteWater()

    print("Filtering chains...")
    remove_chains = []

    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        if not any(
            r.name in PROTEIN_RESIDUES
            for r in residues
        ):
            remove_chains.append(chain)

    if remove_chains:
        print("Removing chains:", [c.id for c in remove_chains])
        modeller.delete(remove_chains)
    print("Chains remaining:", [c.id for c in modeller.topology.chains()])

    print("Removing hydrogens...")
    hydrogen_atoms = []
    for atom in modeller.topology.atoms():
        if atom.element is not None:
            if atom.element.symbol == "H":
                hydrogen_atoms.append(atom)
    modeller.delete(hydrogen_atoms)

    print("Writing clean PDB...")
    with open(output_file, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

def run_pdb2pqr(pdb, pqr):
    print("\nRunning PDB2PQR...")
    run([
        "pdb2pqr",
        "--ff=AMBER",
        "--with-ph=5.0",
        str(pdb),
        str(pqr)
    ])

def write_apbs_input(pqr, outfile, dx_file):
    text = f"""
read
    mol pqr {pqr}
end

elec
    mg-auto
    mol 1
    dime 129 129 129
    cglen 200 200 200
    fglen 120 120 120
    cgcent mol 1
    fgcent mol 1
    lpbe
    temp 298.15
    sdie 78.5
    pdie 2.0
    srfm mol
    srad 1.4
    swin 0.3
    chgm spl2
    sdens 10.0
    bcfl sdh
    ion charge 1 conc 0.150 radius 2.0
    ion charge -1 conc 0.150 radius 2.0
    calcenergy total
    write pot dx {dx_file}
end
quit
"""
    outfile.write_text(text)

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    input_dir = out / "input"
    pqr_dir = out / "pqr"

    input_dir.mkdir(parents=True, exist_ok=True)
    pqr_dir.mkdir(exist_ok=True)
    clean_pdb = (input_dir / "biological_noH.pdb")
    pqr = (pqr_dir / "biological.pqr")
    clean_structure(Path(args.pdb), clean_pdb)
    run_pdb2pqr(clean_pdb, pqr)
    apbs_input = (out / "apbs.in")
    dx_output = (out / "electrostatics")
    write_apbs_input(pqr, apbs_input, dx_output)

    print("\nRunning APBS...")
    run(["apbs", str(apbs_input)])
    print("\nAPBS completed successfully.")

if __name__ == "__main__":
    main()