#!/usr/bin/env python3
import argparse
import os
import openmm
from openmm import app, unit

def get_energy(context):
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )

def main():
    parser = argparse.ArgumentParser(
        description="AMBER parameterization and OpenMM energy minimization."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-iterations", type=int, default=5000)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    output = os.path.join(args.output_dir,f"{base}_minimized.pdb",)
    pdb = app.PDBFile(args.input)

    # AMBER14 protein force field.
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
    initial_energy = get_energy(simulation.context)
    print(f"Initial potential energy: " f"{initial_energy:.3f} kJ/mol")

    simulation.minimizeEnergy(maxIterations=args.max_iterations)
    final_energy = get_energy(simulation.context)
    print(f"Minimized potential energy: " f"{final_energy:.3f} kJ/mol")

    state = simulation.context.getState(getPositions=True)

    with open(output, "w") as handle:
        app.PDBFile.writeFile(
            pdb.topology,
            state.getPositions(),
            handle,
        )

    energy_file = os.path.join(args.output_dir, f"{base}_minimization_energy.txt",)

    with open(energy_file, "w") as handle:
        handle.write(f"Initial_Potential_Energy_kJ_mol,{initial_energy}\n")
        handle.write(f"Final_Potential_Energy_kJ_mol,{final_energy}\n")
        handle.write(f"Max_Minimization_Iterations,{args.max_iterations}\n")
        handle.write("ForceField,AMBER14\n")
        handle.write("WaterModel,TIP3P\n")
        handle.write("Electrostatics,PME\n")
        handle.write("NonbondedCutoff_nm,1.0\n")
    print(f"Minimized structure: {output}")

if __name__ == "__main__":
    main()