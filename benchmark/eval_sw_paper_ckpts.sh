#!/usr/bin/env bash
# Evaluate the paper Shallow Water checkpoints (PAPER_CKPT_MAPPING_DR_SW.md) with
# spectral / gradient metrics via benchmark/eval_checkpoint.py.
#
# Dispatch mode (no args): spreads all six jobs over $GPUS as parallel streams.
#   bash benchmark/eval_sw_paper_ckpts.sh
#   GPUS=4,5,6,7 bash benchmark/eval_sw_paper_ckpts.sh
#
# Worker mode (job names as args): runs them sequentially on the current
# CUDA_VISIBLE_DEVICES. Job names: fno_van fno_act unet_van unet_act trans_van trans_act
#   CUDA_VISIBLE_DEVICES=4 bash benchmark/eval_sw_paper_ckpts.sh fno_act
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${EXP:-$ROOT/experiments}"
DATA_ROOT="${DATA_ROOT:-/path/to/well_data}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/sw_paper_eval}"
GPUS="${GPUS:-4,5,6,7}"
ROLLOUT_BS="${ROLLOUT_BS:-8}"
mkdir -p "$OUT_DIR"

# --batch-size 32 matches the paper runs' data.batch_size, so the single-step test NRMSE
# is comparable to the numbers recorded in PAPER_CKPT_MAPPING_DR_SW.md.
COMMON=(
  benchmark/eval_checkpoint.py
  --dataset shallow_water
  --data-root "$DATA_ROOT"
  --n-steps-input 4
  --batch-size 32
  --data-workers 4
  --rollout-batch-size "$ROLLOUT_BS"
  --spectral-gradient
  --domain-length 5.0
)

# Per-job model flags. Hyperparameters come from each run's extended_config.yaml.
job_args() {
  case "$1" in
    fno_van)
      echo "--ckpt $EXP/shallow_water-deformer_fno_no_deform-DeformerFNO-0.0005/20260428_234640/checkpoints/best.pt --model ACTFNO --no-act"
      ;;
    fno_act)
      echo "--ckpt $EXP/shallow_water-deformer_fno-DeformerFNO-0.0005/20260428_163632/checkpoints/best.pt --model ACTFNO --use-act --max-disp 0.5 --remap-deformers"
      ;;
    unet_van)
      echo "--ckpt $EXP/shallow_water-unet_convnext-UNetConvNext-0.0005/20260502_041256/checkpoints/best.pt --model UNetConvNext --no-act"
      ;;
    unet_act)
      echo "--ckpt $EXP/shallow_water-unet_convnext_deform-UNetConvNextDeform-0.0005/20260504_215908/checkpoints/best.pt --model ACTUNetConvNext --use-act --max-disp-null"
      ;;
    trans_van)
      echo "--ckpt $EXP/shallow_water-transolver_plus-TransolverPlus-0.0005/20260430_013957/checkpoints/best.pt --model TransolverPlus --no-act"
      ;;
    trans_act)
      echo "--ckpt $EXP/shallow_water-transolver_plus_deform-TransolverPlus-0.0005/20260428_163912/checkpoints/best.pt --model TransolverPlus --use-act --max-disp 0.5 --num-deform-heads 16 --deform-hidden-dim 16"
      ;;
    *)
      echo "UNKNOWN_JOB" ;;
  esac
}

run_one() {
  local name="$1"
  local extra
  extra="$(job_args "$name")"
  if [[ "$extra" == "UNKNOWN_JOB" ]]; then
    echo "unknown job: $name" >&2
    return 1
  fi
  local log="$OUT_DIR/${name}.log"
  echo "[$(date +%H:%M:%S)] start $name (GPU=${CUDA_VISIBLE_DEVICES:-all}) -> $log"
  ( cd "$ROOT" && CONDA_NO_PLUGINS=true CUDA_DEVICE_ORDER=PCI_BUS_ID \
      conda run --no-capture-output -n thewell python "${COMMON[@]}" $extra ) \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name (exit $?)"
}

if [[ $# -gt 0 ]]; then
  for job in "$@"; do run_one "$job"; done
  exit 0
fi

# Dispatch mode: Transolver is the slowest, so it gets a GPU each; the faster
# backbones share the remaining GPUs two jobs at a time.
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
STREAMS=(
  "trans_act"
  "trans_van"
  "unet_act unet_van"
  "fno_act fno_van"
)

pids=()
for i in "${!STREAMS[@]}"; do
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  echo "stream $i on GPU $gpu: ${STREAMS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$0" ${STREAMS[$i]} &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "All streams finished (fail=$fail). Logs in $OUT_DIR"
exit "$fail"
