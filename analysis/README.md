## Experiment 0 — Evaluating Graph-pMHC Performance

Experiment 0 characterizes where and why Graph-pMHC performance varies across HLA-II allotypes. It establishes per-allotype, allele, heterodimer, and MHC-class performance baselines, then tests whether poor performance is associated with pseudosequence divergence, training density, high-confidence prediction errors, or isolation in MHC sequence space.

Together, these analyses distinguish potential **data-coverage effects from limitations of the current graph representation**, providing the rationale for subsequent structural and molecular-dynamics analyses.





Preliminary hypotheses included expanding adjacency matrices cutoff of 4angstroms (hard contacts) to 8angstroms (hard+soft contacts; second-shell interactions) to extract continuous distances maps from AlphaFold2-Multimer structures which are otherwise discarded in the default binary classification, motivating a distance-weighted edge representation (i.e. radial basis function-esque) as a principled improvement. ***requires generating 115 AF2 structures to reproduce original workflow, on hold pending molecular dynamics pilot results (see analysis)


