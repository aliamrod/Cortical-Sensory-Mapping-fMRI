#!/bin/bash
set -euo pipefail

# Sensory Feature Extraction (fsaverage5) -- SLURM Array Launcher + Worker
# Objective(s)
# This script runs large-scale feature extraction across all runs in manifest CSV (./master-227.csv) utilizing SLURM array jobs. 

# -----------------------------
# PATHS
# -----------------------------
MASTER=/home/amahama/PROJECTS/1_sensory/data/manifests/master-227.csv
TEMPLATES=/home/amahama/PROJECTS/1_sensory/data/manifests/templates
OUTDIR=/home/amahama/PROJECTS/1_sensory/data/manifests/workdir/sensory_features
SCRIPT=/home/amahama/PROJECTS/1_sensory/data/manifests/2_SCRIPTS/2_extract_features_fsavg5.py

CHUNK_SIZE=50
MAX_PARALLEL=20

mkdir -p logs
mkdir -p "${OUTDIR}"

# -----------------------------
# IF NOT IN SLURM: SUBMIT ARRAY
# -----------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then

    echo "Counting rows..."

    N_ROWS=$(python3 - <<PY
import pandas as pd
print(len(pd.read_csv("${MASTER}")))
PY
)

    N_CHUNKS=$(( (N_ROWS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    MAX_IDX=$(( N_CHUNKS - 1 ))

    echo "Rows:        ${N_ROWS}"
    echo "Chunk size:  ${CHUNK_SIZE}"
    echo "Chunks:      ${N_CHUNKS}"
    echo "Submitting:  0-${MAX_IDX}%${MAX_PARALLEL}"

    sbatch \
      --job-name=sensory_feat \
      --output=logs/sensory_feat_%A_%a.out \
      --error=logs/sensory_feat_%A_%a.err \
      --time=12:00:00 \
      --cpus-per-task=4 \
      --mem=16G \
      --array=0-${MAX_IDX}%${MAX_PARALLEL} \
      "$0"

    exit 0
fi

# -----------------------------
# SLURM WORKER
# -----------------------------
echo "Running chunk ${SLURM_ARRAY_TASK_ID}"

python3 "${SCRIPT}" \
  --master-csv "${MASTER}" \
  --templates-dir "${TEMPLATES}" \
  --outdir "${OUTDIR}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-idx "${SLURM_ARRAY_TASK_ID}" \
  --require-ok \
  --write-hsv \
  --skip-existing

echo "Done chunk ${SLURM_ARRAY_TASK_ID}"
