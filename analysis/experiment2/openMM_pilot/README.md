## OpenMM Pilot ##
`run_analysis.sh` to execute
Representative structure: DQ2.5 + acidic peptide (electrostatics test case)

### Workflow: ###
1 - Protonation & Environment Initialization
  - `PDB2PQR` `PROPKA` protonation at pH 5.0 (AF2m does not model H+ bonds & pMHC binds in endosome)

2 - Geometric & Solvent Surface Profiling
  - `FreeSASA` `MDTraj` to calculate solvent-accessible surface area (SASA)    
  
3 - Physics Refinement by Gradient Descent
  - `AMBER` force-field parameterization and localized energy minimization

4 - Physics Extraction
  - 3D coordinates and forcefield-assigned partial charges
  - Global energy (total potential energy & isolated energy)
  - Residual atomic forces (local structural strain)

Pilot tests whether physically-informed protonation and molecular mechanics provide useful
electrostatic representations that can complement Gpmhc representation as a tensor overlap (node feature fusion and edge feature inclusion). 
