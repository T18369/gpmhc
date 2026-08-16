## Experiment 0 — Evaluating Graph-pMHC Performance
Experiment 0 characterizes where and why Graph-pMHC performance varies across HLA-II allotypes. It establishes per-allotype, allele, heterodimer, and MHC-class performance baselines, then tests whether poor performance is associated with pseudosequence divergence, training density, high-confidence prediction errors, or isolation in MHC sequence space.

Together, these analyses distinguish data-coverage effects from limitations of the current graph representation, providing the rationale for subsequent structural and molecular-dynamics analyses.

## Experiment 1 - Structural Analysis ##
Experiment 1 selects informative peptide-MHCII examples, generates FASTA inputs for structure 
predictions, and extracts peptide-MHC distance maps from predicted structures, with additional analysis on contact ratios (hard-proximal-distal). AlphaFold-Multimer structures are generated through ColabFold, and results are processed to generate continuous and binary interface representations. 

*Preliminary hypotheses included expanding adjacency matrices cutoff of 4 Angstroms (hard contacts) to 8 Angstroms (hard+soft contacts; second-shell interactions) to extract continuous distances maps from AF2m structures which are otherwise discarded in the default binary classification, motivating a distance-weighted edge representation (i.e. radial basis function) as a principled improvement. This is particularly relevant for poor performing DQ alleles 2.5 and 8, wherein pocket interactions are dependent on long-range electrostatic "steering"*

## Experiment 2 - Molecular Dynamics Simulations ##
Experiment 2 builds on findings from Experiment 1, wherein poor-performing DQ alleles that inherently depend on negative electrostatic force fields to engage acidic peptides are modeled. 
