#!/usr/bin/env python3
import argparse
import os
import openmm
from openmm import app, unit
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Extract coordinates, charges, energies, and atomic forces."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    pdb = app.PDBFile(args.input)

    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml",)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    simulation = app.Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
    )
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(
        getPositions=True,
        getEnergy=True,
        getForces=True,
    )

    positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )

    total_energy = state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

    # Extract AMBER partial charges.
    nonbonded = None

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded = force
            break

    if nonbonded is None:
        raise RuntimeError(
            "Could not find OpenMM NonbondedForce."
        )

    rows = []

    for atom in pdb.topology.atoms():
        idx = atom.index

        charge, _, _ = nonbonded.getParticleParameters(idx)

        residue = atom.residue
        
        if residue.name in {"NA", "CL"}:
            component = "ion"
        elif residue.chain.id == "D":
            component = "water"
        elif residue.chain.id in {"A", "B"}:
            component = "mhc"
        elif residue.chain.id == "C":
            component = "peptide"
        else:
            component = "other"

        rows.append(
            {
                "atom_idx": idx,
                "atom_name": atom.name,
                "residue": residue.name,
                "residue_idx": residue.index,
                "residue_number": residue.id,
                "chain": residue.chain.id,
                "x_A": positions[idx][0],
                "y_A": positions[idx][1],
                "z_A": positions[idx][2],
                "partial_charge_e": charge.value_in_unit(
                    unit.elementary_charge
                ),
                "force_x_kJ_mol_nm": forces[idx][0],
                "force_y_kJ_mol_nm": forces[idx][1],
                "force_z_kJ_mol_nm": forces[idx][2],
                "force_magnitude_kJ_mol_nm": (
                    forces[idx][0] ** 2
                    + forces[idx][1] ** 2
                    + forces[idx][2] ** 2
                ) ** 0.5,
                "component": component,
            }
        )

    df = pd.DataFrame(rows)
    force_summary = (
        df.groupby("component")["force_magnitude_kJ_mol_nm"]
        .agg(
            atom_count="count",
            mean_force="mean",
            median_force="median",
            max_force="max",
        )
        .reset_index()
    )
    
    force_summary.to_csv(
        os.path.join(args.output_dir, f"{base}_force_summary.csv",),
        index=False,
    )
    


    # Summarize residual forces for the biological pMHC system only.
    # Excludes water and ions added during TIP3P solvation.
    biological_df = df[df["component"].isin(["mhc", "peptide"])]
    
    bio_summary = (
        biological_df["force_magnitude_kJ_mol_nm"]
        .describe(percentiles=[0.95, 0.99])
    )
    print("Biological system force summary:")
    print(bio_summary)
    bio_output = os.path.join(args.output_dir, f"{base}_biological_force_summary.csv",)
    bio_summary.to_csv(bio_output,)
    biological_force = biological_df["force_magnitude_kJ_mol_nm"]
    max_force = biological_force.max()
    mean_force = biological_force.mean()


    for component in ["mhc", "peptide"]:
        subset = df[df["component"] == component]
        print(component)
        print(subset["force_magnitude_kJ_mol_nm"].describe(percentiles=[0.95,0.99]))
    
    atom_output = os.path.join(args.output_dir, f"{base}_atom_features.csv",)
    energy_output = os.path.join(args.output_dir, f"{base}_energy.csv",)
    df.to_csv(atom_output, index=False)

    pd.DataFrame(
        [
            {
                "structure": base,
                "total_potential_energy_kJ_mol": total_energy,
                "mean_biological_force_kJ_mol_nm": mean_force,
                "max_biological_force_kJ_mol_nm": max_force,
            }
        ]
    ).to_csv(energy_output, index=False,)

    print(f"Total potential energy: {total_energy:.3f} kJ/mol")
    print(f"Atom features: {atom_output}")
    print(f"Energy table: {energy_output}")

if __name__ == "__main__":
    main()