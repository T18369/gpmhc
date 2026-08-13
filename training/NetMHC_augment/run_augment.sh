#!/bin/bash
set -e
mkdir -p outputs

echo "Running NetMHCIIpan DQ augmentation pipeline"

python3 01_extract_dq_heterodimers.py
python3 02_extract_el_heterodimer_pairs.py
python3 03_filter_el_redundancy.py
python3 04_exact_overlap_filter.py
python3 05_kmer_similarity_screen.py
python3 06_filter_kmer_leakage_balance.py
python3 07_cdhit_sequence_similarity.py
python3 08_merge_el_augmentation.py
python3 09_candidate_peptides.py
python3 10_generate_DQ_window_augmented.py
python3 11_append_DQ_to_HLAII_train.py

echo "DQ augmentation pipeline complete"
