#!/usr/bin/env bash
# Launch all 8 Poisson-Gauss training jobs, one per GPU.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$ROOT/benchmark"
LOG_DIR="$ROOT/experiments/poisson_gauss_train_logs"
mkdir -p "$LOG_DIR"

EXPERIMENTS=(
  "act_fno_poisson"
  "vanilla_fno_poisson"
  "act_convnext_poisson"
  "vanilla_unet_convnext_poisson"
  "act_transolver_poisson"
  "vanilla_transolver_poisson"
  "flower_poisson"
  "adrnet_poisson"
)

pids=()
for i in "${!EXPERIMENTS[@]}"; do
  exp="${EXPERIMENTS[$i]}"
  gpu=$i
  log="$LOG_DIR/${exp}.log"
  echo "[$(date +%H:%M:%S)] GPU $gpu: $exp -> $log"
  (
    CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CONDA_NO_PLUGINS=true WANDB_MODE=disabled \
    conda run --no-capture-output -n thewell \
      python "$BENCH/train.py" \
        experiment="$exp" \
        server=local \
        "data.well_base_path=${DATA_ROOT:-/path/to/well_data}" \
        trainer.epochs=300
  ) > "$log" 2>&1 &
  pids+=($!)
done

echo "All 8 jobs launched. Waiting..."
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "[$(date +%H:%M:%S)] All done (fail=$fail). Logs: $LOG_DIR"
exit "$fail"
