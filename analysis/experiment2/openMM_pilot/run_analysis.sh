#!/bin/bash
set -e

INPUT="analysis/experiment2/openmm_pilot/input/DQ25_DQA1-05-05_DQB1-03-01_DVQDDEEREL.pdb"

NAME=$(basename "$INPUT" .pdb)
BASE="analysis/experiment2/openmm_pilot/analysis/$NAME"

mkdir -p "$BASE"

echo "Running OpenMM pilot: $NAME"

echo "1. Protonation and environment initialization"
python3 analysis/experiment2/openmm_pilot/01_protonate.py \
    --input "$INPUT" \
    --output-dir "$BASE/protonated" \
    --ph 5.0

echo "2. SASA profiling"
python3 analysis/experiment2/openmm_pilot/02_sasa.py \
    --input "$BASE/protonated/${NAME}_protonated.pdb" \
    --output-dir "$BASE/sasa"

echo "3. Solvation"
python3 analysis/experiment2/openmm_pilot/03_solvate.py \
    --input "$BASE/protonated/${NAME}_protonated.pdb" \
    --output "$BASE/solvated/${NAME}_solvated.pdb"

echo "4. Energy minimization"
python3 analysis/experiment2/openmm_pilot/04_minimize.py \
    --input "$BASE/solvated/${NAME}_solvated.pdb" \
    --output-dir "$BASE/minimized"

echo "5. Physics extraction"
python3 analysis/experiment2/openmm_pilot/05_extract_physics.py \
    --input "$BASE/minimized/${NAME}_solvated_minimized.pdb" \
    --output-dir "$BASE/extracted"

echo "OpenMM pilot complete: $BASE"