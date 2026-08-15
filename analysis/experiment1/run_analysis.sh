#!/bin/bash
set -e
cd gpmhc/analysis/experiment1

echo "Running DQ structural analysis pipeline"

python3 01_select_structures.py
python3 02_generate_fasta.py
python3 03_extract_distance_maps.py
python3 04_contact_architecture.py

echo "Pipeline complete"
