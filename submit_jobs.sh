#!/usr/bin/env bash
# submit_jobs.sh — runs DIP, PIP, MCP groups in parallel (3 GPUs)
#
# Usage:
#   bash submit_jobs.sh --zip /path/to/dataset.zip
#   bash submit_jobs.sh --zip /path/to/dataset.zip --fast
#   bash submit_jobs.sh --resume --skip-baseline

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ZIP_PATH=""
DATA_ROOT="./data"
EPOCHS_GEN=300
EPOCHS_CLF=50
RUNS=3
N_SYNTH=1000
RUN_DDPM=0
RESUME=""
SKIP_BASELINE=0
SKIP_CLF=0

GROUPS=(dip pip mcp)

# Auto-detect available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
echo "  Detected $NUM_GPUS GPU(s)"

# Assign GPUs to groups
if [[ $NUM_GPUS -ge 3 ]]; then
  GPU_MAP=(0 1 2)
  echo "  Layout: DIP→GPU0  PIP→GPU1  MCP→GPU2 (all parallel)"
elif [[ $NUM_GPUS -eq 2 ]]; then
  GPU_MAP=(0 1 0)
  echo "  Layout: DIP→GPU0  PIP→GPU1  MCP→GPU0 after DIP"
else
  GPU_MAP=(0 0 0)
  echo "  Layout: all groups sequential on GPU0"
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --zip)        ZIP_PATH="$2";      shift 2 ;;
    --data_root)  DATA_ROOT="$2";     shift 2 ;;
    --epochs_gen) EPOCHS_GEN="$2";   shift 2 ;;
    --epochs_clf) EPOCHS_CLF="$2";   shift 2 ;;
    --runs)       RUNS="$2";          shift 2 ;;
    --ddpm)       RUN_DDPM=1;         shift ;;
    --resume)     RESUME="--resume";  shift ;;
    --skip-baseline) SKIP_BASELINE=1; shift ;;
    --skip-clf)   SKIP_CLF=1;         shift ;;
    --fast)       EPOCHS_GEN=5; EPOCHS_CLF=5; RUNS=1; N_SYNTH=20; shift ;;
    *)            echo "Unknown: $1"; exit 1 ;;
  esac
done

# STEP 0: Extract zip
if [[ -n "$ZIP_PATH" ]]; then
  if [[ ! -f "$DATA_ROOT/metadata.csv" ]]; then
    echo "── Extracting dataset (login node) ──────────────────────────────"
    python3 setup_data.py --zip "$ZIP_PATH" --data_root "$DATA_ROOT"
  else
    echo "  metadata.csv exists — skipping extraction."
  fi
fi

if [[ ! -f "$DATA_ROOT/metadata.csv" ]]; then
  echo "[ERROR] $DATA_ROOT/metadata.csv missing."
  exit 1
fi

# Preserve existing outputs and logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -d "outputs" ]]; then
  cp -r outputs outputs_backup_$TIMESTAMP
  echo "  Backed up outputs → outputs_backup_$TIMESTAMP"
fi
if [[ -d "logs" ]]; then
  cp -r logs logs_backup_$TIMESTAMP
  echo "  Backed up logs → logs_backup_$TIMESTAMP"
fi

MODELS="cyclegan wgan_gp cvae"
[[ $RUN_DDPM -eq 1 ]] && MODELS="$MODELS ddpm"

mkdir -p logs

echo "================================================================"
echo "  3-GPU group run: DIP (GPU 0) · PIP (GPU 1) · MCP (GPU 2)"
echo "  Gen epochs: $EPOCHS_GEN  |  CLF epochs: $EPOCHS_CLF  |  Runs: $RUNS"
echo "  Models: $MODELS  |  Aug ratios: 0.3, 0.5, 1.0"
echo "  $(date)"
echo "================================================================"

# STEP 1: Baseline per group + group workers all in parallel
echo ""
PIDS=()
BASELINE_PIDS=()

if [[ $SKIP_BASELINE -eq 0 ]]; then
  echo "── Launching baselines (pooled + per-group) ─────────────────────"

  # Pooled baseline on GPU 0
  CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py \
    --epochs "$EPOCHS_CLF" --runs "$RUNS" \
    >> logs/baseline_pooled.out 2>&1 &
  BASELINE_PIDS+=($!)
  echo "  Baseline pooled  (pid ${BASELINE_PIDS[-1]},  tail -f logs/baseline_pooled.out)"

  # Per-group baselines
  CUDA_VISIBLE_DEVICES=0 python3 train_baseline.py --joint dip \
    --epochs "$EPOCHS_CLF" --runs "$RUNS" \
    >> logs/baseline_dip.out 2>&1 &
  BASELINE_PIDS+=($!)
  echo "  Baseline DIP     (pid ${BASELINE_PIDS[-1]},  tail -f logs/baseline_dip.out)"

  CUDA_VISIBLE_DEVICES=1 python3 train_baseline.py --joint pip \
    --epochs "$EPOCHS_CLF" --runs "$RUNS" \
    >> logs/baseline_pip.out 2>&1 &
  BASELINE_PIDS+=($!)
  echo "  Baseline PIP     (pid ${BASELINE_PIDS[-1]},  tail -f logs/baseline_pip.out)"

  CUDA_VISIBLE_DEVICES=1 python3 train_baseline.py --joint mcp \
    --epochs "$EPOCHS_CLF" --runs "$RUNS" \
    >> logs/baseline_mcp.out 2>&1 &
  BASELINE_PIDS+=($!)
  echo "  Baseline MCP     (pid ${BASELINE_PIDS[-1]},  tail -f logs/baseline_mcp.out)"
else
  echo "── Baseline skipped (--skip-baseline) ───────────────────────────"
fi

# STEP 2: Launch 3 GPU workers, one per joint group
echo ""
echo "── Launching 3 group workers ────────────────────────────────────"

run_group() {
  local GROUP=$1
  local GPU=$2
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "=== GROUP $GROUP on GPU $GPU started $(date) ===" >> "logs/group_${GROUP}.out"

  python3 tune_models.py --joint "$GROUP" --models cyclegan wgan_gp cvae --gpu 0 \
    >> "logs/group_${GROUP}.out" 2>&1

  python3 train_cyclegan.py --joint "$GROUP" --epochs "$EPOCHS_GEN" --pair 1,3 $RESUME \
    >> "logs/group_${GROUP}.out" 2>&1
  python3 train_cyclegan.py --joint "$GROUP" --epochs "$EPOCHS_GEN" --pair 2,4 $RESUME \
    >> "logs/group_${GROUP}.out" 2>&1
  python3 train_wgan_gp.py  --joint "$GROUP" --epochs "$EPOCHS_GEN" $RESUME \
    >> "logs/group_${GROUP}.out" 2>&1
  python3 train_cvae.py     --joint "$GROUP" --epochs "$EPOCHS_GEN" $RESUME \
    >> "logs/group_${GROUP}.out" 2>&1

  if [[ $RUN_DDPM -eq 1 ]]; then
    python3 train_ddpm.py --joint "$GROUP" --epochs 150 $RESUME \
      >> "logs/group_${GROUP}.out" 2>&1
  fi

  if [[ $SKIP_CLF -eq 0 ]]; then
    python3 generate_samples.py --joint "$GROUP" --models $MODELS --n "$N_SYNTH" \
      >> "logs/group_${GROUP}.out" 2>&1
    python3 train_augmented.py  --joint "$GROUP" \
      --epochs "$EPOCHS_CLF" --runs "$RUNS" --models $MODELS \
      >> "logs/group_${GROUP}.out" 2>&1
  fi

  echo "=== GROUP $GROUP on GPU $GPU finished $(date) ===" >> "logs/group_${GROUP}.out"
}

# Launch groups
PIDS=()
echo ""
echo "── Launching group workers ──────────────────────────────────────"

run_group dip 0 &
PIDS+=($!)
echo "  GPU 0 → dip  (pid ${PIDS[-1]},  tail -f logs/group_dip.out)"

run_group pip 1 &
PIDS+=($!)
echo "  GPU 1 → pip  (pid ${PIDS[-1]},  tail -f logs/group_pip.out)"

# MCP waits for DIP if sharing GPU
if [[ $NUM_GPUS -ge 3 ]]; then
  run_group mcp 2 &
  PIDS+=($!)
  echo "  GPU 2 → mcp  (pid ${PIDS[-1]},  tail -f logs/group_mcp.out)"
else
  echo "  GPU 0 → mcp  (will start after dip finishes)"
  DIP_PID="${PIDS[0]}"
  (
    while kill -0 $DIP_PID 2>/dev/null; do sleep 10; done
    run_group mcp 0
  ) &
  PIDS+=($!)
fi

# Wait for all group workers
echo ""
echo "── Waiting for group workers ─────────────────────────────────────"
FAILED=0
WAIT_GROUPS=(dip pip mcp)
for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  GROUP="${WAIT_GROUPS[$i]}"
  if wait "$PID"; then
    echo "  ✓ $GROUP done"
  else
    echo "  ✗ $GROUP FAILED (check logs/group_${GROUP}.out)"
    FAILED=$((FAILED + 1))
  fi
done

# Wait for baselines
if [[ ${#BASELINE_PIDS[@]} -gt 0 ]]; then
  echo ""
  echo "── Waiting for baselines ────────────────────────────────────────"
  BASE_GROUPS=(pooled dip pip mcp)
  for i in "${!BASELINE_PIDS[@]}"; do
    GROUP="${BASE_GROUPS[$i]}"
    if wait "${BASELINE_PIDS[$i]}"; then
      echo "  ✓ baseline $GROUP done"
    else
      echo "  ✗ baseline $GROUP FAILED (check logs/baseline_${GROUP}.out)"
      FAILED=$((FAILED + 1))
    fi
  done
fi

# STEP 3: Evaluate
echo ""
echo "── Evaluation ───────────────────────────────────────────────────"
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py 2>&1 | tee logs/evaluate.out

# STEP 4: Visualize
echo ""
echo "── Generating figures ───────────────────────────────────────────"
python3 visualize_results.py 2>&1 | tee logs/visualize.out

echo ""
echo "================================================================"
echo "  DONE  $(date)"
echo "  Results:  outputs/results/"
echo "  Figures:  outputs/figures/"
echo "================================================================"