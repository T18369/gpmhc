#!/bin/bash
set -e

mkdir -p results/figures
mkdir -p results/tables

echo "Running gpmhc qc analysis"
python3 02b_ap_by_allotype_test_split.py
python3 02b_ap_by_mhc_class_test.py
python3 02c_netmhcpan_ap_by_allotype_test_split.py
python3 02d_compare_graphpmhc_vs_netmhcpan_ap.py
python3 02e_ap_by_individual_allele_test_split.py
python3 02f_ap_by_heterodimer_test_split.py
python3 03_divergence_vs_ap_test_split.py
python3 04_confound_control_test_split.py
python3 05_confident_errors_test_split.py
python3 08_mhc_similarity_error_test_split.py

echo "Analysis complete"
echo "Figures: results/figures/"
echo "Tables: results/tables/"
