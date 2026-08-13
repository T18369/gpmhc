## NetMHCIIpan DQ Training Augmentation ##
To address the lower performance observed for HLA-DQ, a targeted augmentation of the training set was
performed using DQ peptides from the NetMHCIIpan 4.2 training data. The objective is to enhance
coverage by increasing the number of sequence-diversified training examples

*Scatter plot depicts allele-specific performance following additional training*

### Pipeline (Filtering/preprocessing) ###
1 - DQ heterodimer characterization `01_parse_dq_heterodimers.py` `02_extract_el_heterodimer_pairs.py`
only 5 exactly paired DQ heterodimers shared across both NetMHC and Graph-pMHC

2 - EL+/- extraction `03_filter_el_redundancy.py` Produced ~23k EL+ (mean seq. overlap = 0.571) and 
354k EL- (mean seq. overlap = 0.000149; randomly generated)

3 - Filter duplicates `04_exact_overlap_filter.py` Prevent inflation of pre-existing training data

4 - K-mer similarity screen `05_kmer_similarity_screen.py` `06_filter_kmer_leakage.py` `07_cdhit_sequence_similarity.py` (diagnostic) `08_merge_el_augmentation.py`
25k DQ examples across 27 DQ alleles

5 - 9mer core filter `09_candidate_peptides.py` `10_generate_DQ_window_augmented.py`
19+mer peptides enumerated into 9mer sliding windows with 5aa N & C terminal flanks, retaining valid cores

6 - Assembly `11_append_DQ_to_HLAII_train.py`
Final augmentation contributed ~9.4k unique DQ training examples. Appended to training set.
Did not filter according to sequence similarity with gpmhc beyond exact duplicates given the small
proportion of samples within the overall training set. 

### Result ###
Augmentation did not produce meaningful improvement in aggregate test performance relative to het-only
baseline (DQ test AP =~ 0.779); suggests DQ performance bottleneck is unlikely to be explained
by insufficient training data. More likely explanation is the representation of sequence-structure
relationships, unlikely to be resolved by adding more training samples. 

### Note ###
DQ haplotypes that perform poorly are DQ2.5 and DQ8; heavily associated with T1D susceptibility (>80%
of all patients) and Celiac disease (>95%). Both feature highly-positive anchor pockets creating an
electrostatic field that binds negatively-charged processed gluten peptides and insulin beta-chain peptides.
This trait helps bind negatively-charged proteins from pathogens like cholera & TB; i.e. evolutionary trade-off.
**This electrostatic field is likely ignored in the 4angstrom cutoff adjacency matrix, rationalizing 
continuous edge mapping with chemistry annotations.**
