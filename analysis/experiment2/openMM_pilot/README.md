## OpenMM Pilot ##
`run_analysis.sh` to execute
Representative structure: DQ2.5 + acidic peptide (electrostatics test case)

Workflow:
  - PDB2PQR/PROPKA protonation at pH 5.0 (AF2m does not model H+ bonds & pMHC binds in endosome)
  - Force-field parameterization
  - Energy minimization (gradient descent)
  - 3D coordinates and partial charges
  - Electrostatic input for graph

Pilot tests whether physically-informed protonation and molecular mechanics provide useful
electrostatic representations that can complement Gpmhc representation. 
