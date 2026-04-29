#!/usr/bin/env bash
# run_pipeline.sh — sequential per-joint pipeline (single GPU, VSCode terminal on SCC)
#
# For maximum speed, use submit_jobs.sh instead (SLURM array, 12 joints in parallel).
# This script is the fallback if you want to run everything from one VSCode terminal.
#
# Usage:
#   bash run_pipeline.sh --zip /path/to/dataset.zip
#   bash run_pipeline.sh --zip /path/to/dataset.zip --fast
#   bash run_pipeline.sh --zip /path/to/dataset.zip --joints "dip2,pip2,mcp2"
#   bash run_pipeline.sh --zip /path/to/dataset.zip --resume
#   bash run_pipeline.sh --zip /path/to/dataset.zip --pool   # skip per-joint, pool all
#
# With --pool: ~7-8hr on A100
# With --joints (subset): proportionally faster
# All 12 joints sequential: ~36-48hr — use submit_jobs.sh instead

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_PATH=""
DATA_ROOT="./data"
EPOCHS_GEN=100
EPOCHS_CLF=50
RUNS=3
N_SYNTH=1000
RESUME=""
POOL=0
FAST=0
RUN_DDPM=0
JOINTS=""

ALL_JOINTS="dip2 dip3 dip4 dip5 pip2 pip3 pip4 pip5 mcp2 mcp3 mcp4 mcp5"

while [[ $# -gt 0 ]]; do
  case $1 in
    --zip)        ZIP_PATH="$2";      shift 2 ;;
    --data_root)  DATA_ROOT="$2";     shift 2 ;;
    --epochs_gen) EPOCHS_GEN="$2";   shift 2 ;;
    --epochs_clf) EPOCHS_CLF="$2";   shift 2 ;;
    --runs)       RUNS="$2";          shift 2 ;;
    --joints)     JOINTS="$2";        shift 2 ;;
    --resume)     RESUME="--resume";  shift ;;
    --pool)       POOL=1;             shift ;;
    --fast)       FAST=1; EPOCHS_GEN=5; EPOCHS_CLF=5; RUNS=1; N_SYNTH=20; shift ;;
    --ddpm)       RUN_DDPM=1;         shift ;;
    *)            echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if conda info --envs 2>/dev/null | grep -q "oa_aug"; then
  source activate oa_aug 2>/dev/null || conda activate oa_aug
fi

# STEP 0: Extract zip
if [[ -n "$ZIP_PATH" ]]; then
  echo ""
  echo "── STEP 0: Extracting dataset ───────────────────────────────────"
  python setup_data.py --zip "$ZIP_PATH" --data_root "$DATA_ROOT"
fi

if [[ ! -f "$DATA_ROOT/metadata.csv" ]]; then
  echo "[ERROR] $DATA_ROOT/metadata.csv not found."
  echo "Run: python setup_data.py --zip /path/to/data.zip"
  exit 1
fi

if [[ $POOL -eq 1 ]]; then
  JOINT_LIST="pooled"
  echo "Mode: POOLED (all joints combined)"
elif [[ -n "$JOINTS" ]]; then
  JOINT_LIST="${JOINTS//,/ }"
  echo "Mode: per-joint subset: $JOINT_LIST"
else
  JOINT_LIST="$ALL_JOINTS"
  echo "Mode: all 12 joints (sequential — consider submit_jobs.sh for parallel)"
fi

MODELS="cyclegan wgan_gp cvae"
if [[ $RUN_DDPM -eq 1 ]]; then MODELS="$MODELS ddpm"; fi

echo "Gen epochs=$EPOCHS_GEN | CLF epochs=$EPOCHS_CLF | Runs=$RUNS | Models=$MODELS"

# STEP 1: Baseline
echo ""
echo "── STEP 1: Baseline ResNet-18 ───────────────────────────────────"
python train_baseline.py --epochs "$EPOCHS_CLF" --runs "$RUNS"

# Per-joint loop
for JOINT in $JOINT_LIST; do
  echo ""
  echo "══ JOINT: $JOINT ══════════════════════════════════════════════"

  python train_cyclegan.py --joint "$JOINT" --epochs "$EPOCHS_GEN" --pair 1,3 $RESUME
  python train_cyclegan.py --joint "$JOINT" --epochs "$EPOCHS_GEN" --pair 2,4 $RESUME
  python train_wgan_gp.py  --joint "$JOINT" --epochs "$EPOCHS_GEN" $RESUME
  python train_cvae.py     --joint "$JOINT" --epochs "$EPOCHS_GEN" $RESUME
  [[ $RUN_DDPM -eq 1 ]] && python train_ddpm.py --joint "$JOINT" --epochs 100 $RESUME

  python generate_samples.py --joint "$JOINT" --models $MODELS --n "$N_SYNTH"
  python train_augmented.py  --joint "$JOINT" --epochs "$EPOCHS_CLF" --runs "$RUNS" --models $MODELS
done

# STEP 8: Evaluation
echo ""
echo "── STEP 8: Evaluation ───────────────────────────────────────────"
python evaluate.py

echo ""
echo "Done. See outputs/results/comparison_table.csv"
