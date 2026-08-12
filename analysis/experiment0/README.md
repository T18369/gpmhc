## Reproducing qc analysis

To run complete analysis:

```bash
run_analysis.sh
```


## Experiment 0 — Evaluating Graph-pMHC Performance

**Objective:** Determine whether Graph-pMHC performance variation across HLA-II allotypes reflects training coverage, MHC sequence-space divergence, prediction uncertainty, or limitations of the current graph representation.

All analyses use the predefined **test split** unless otherwise noted. Average Precision (AP) is calculated directly from model scores without applying a classification threshold.

### 0.2 — Per-allotype performance atlas
**Scripts:** `02b_ap_by_allotype_test_split.py`, `02b_ap_by_mhc_class_test.py`

Establishes the performance baseline across individual allotypes and MHC classes (DP, DQ, DR), identifying low- and high-performing allotypes.

**Outputs:** `results_test_split/`

### 0.2c — NetMHCIIpan comparison
**Script:** `02c_netmhcpan_ap_by_allotype_test_split.py`

Calculates NetMHCIIpan AP using the same test split and allotype definitions, enabling direct comparison with Graph-pMHC. Percentile ranks are sign-inverted so higher values indicate stronger predictions.

**Outputs:** `results_netmhcpan_test_split/`

### 0.2d — Individual allele analysis
**Script:** `02d_ap_by_individual_allele_test_split.py`

Decomposes composite HLA-II allotypes into constituent α/β alleles and evaluates Graph-pMHC and NetMHCIIpan performance at the individual allele level.

**Outputs:** `results_individual_allele_test/`

### 0.2f — Heterodimer-level analysis
**Script:** `02f_ap_by_heterodimer_test_split.py`

Evaluates performance at the biologically defined MHC-II heterodimer level (DP, DQ, DR), retaining only unambiguous α/β pairings.

**Outputs:** `results_heterodimer_test/`

### 0.3 — Pseudosequence divergence vs performance
**Script:** `03_divergence_vs_ap_test_split.py`

Tests whether poor allotype-level AP is associated with MHC pseudosequence divergence using consensus divergence and nearest-neighbour distance.

**Outputs:** `results/divergence/`

### 0.4 — Training-density confound control
**Script:** `04_confound_control.py`

Tests whether the divergence–AP relationship is explained by reduced training coverage rather than representation limitations using coverage correlation, partial correlation, coverage-stratified analysis, and multiple regression.

**Outputs:** `results/confound/`

### 0.5 — High-confidence errors
**Script:** `05_confident_errors.py`

Identifies extreme-score disagreements with the ground truth: high-confidence false positives (≥90th percentile, EL=0) and false negatives (≤10th percentile, EL=1). This tests whether failures occur at prediction extremes rather than only near an implicit classification boundary.

**Outputs:** `results/confident_errors/`

### 0.8 — MHC similarity vs performance
**Script:** `08_mhc_similarity_error.py`

Tests whether poorly performing allotypes cluster in pseudosequence space or instead occupy relatively isolated regions. High- and low-error groups are defined as the bottom and top 20% by test-set AP.

**Outputs:** `results/mhc_similarity/`

### Overall logic

**Performance atlas → localization of failures → sequence-space association → training-density control → confident-error characterization → sequence-space isolation**

Together, these analyses establish whether Graph-pMHC failures are better explained by **data sparsity or limitations in representing structurally unusual MHC sequence space**, motivating subsequent structural and molecular-dynamics analyses.
