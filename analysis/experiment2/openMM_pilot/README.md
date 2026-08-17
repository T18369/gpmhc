## OpenMM Pilot ##
`run_analysis.sh` to execute sample structure: DQ2.5 + acidic peptide (electrostatics test case)

### Workflow: ###
1 - Protonation & Environment Initialization `01_protonate.py`
  - `PROPKA` protonation at pH 5.0 (AF2m does not model H+ bonds & pMHC binds in endosome). `PDB2PQR` resolves missing terminal oxygen atom warnings. Check for any missing contacts in backbone.

2 - Geometric & Solvent Surface Profiling `02_sasa.py`
  - `FreeSASA` `MDTraj` to calculate solvent-accessible surface area (SASA)

3 - Solvent Environment `03_solvent.py`
  - Add `TIP3P` water, generate aqueous environment
  - Periodic rectangular box
  
4 - Physics Refinement by Gradient Descent `04_minimize.py`
  - `AMBER` force-field parameterization and localized energy minimization

5 - Physics Extraction `05_extract_physics.py`
  - 3D coordinates and forcefield-assigned partial charges
  - Global energy (total potential energy & isolated energy)
  - Residual atomic forces (local structural strain)

6 - Residue-Level Physics `06_residue_physics.py`
  - Atom-level properties into residual features (total/average charge, force magnitude etc.).

7 - Residue-Level Physics Visualization `06_visualize_residue_physics.py`
  - Map physical features back onto the protein structure for inspection

8 - Electrostatic Potential Mapping `08_run_apbs.py`
  - Generate a 3D electrostatic potential field using `APBS` Poisson-Boltzmann (maps long-range charge interactions)

Pilot tests whether physically-informed protonation and molecular mechanics provide useful
electrostatic representations that can complement Gpmhc representation as a tensor overlap (node feature fusion and edge feature inclusion). 
