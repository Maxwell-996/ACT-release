#!/usr/bin/env bash
# Evaluate MHD_64 checkpoints (data/MHD_64_ckpts/ + Flower experiment) and
# SW-ADRNet via benchmark/eval_checkpoint.py.
#
# Dispatch mode (no args): spreads jobs over $GPUS as parallel streams.
#   bash benchmark/eval_mhd64_ckpts.sh
#   GPUS=4,5,6,7 bash benchmark/eval_mhd64_ckpts.sh
#
# Worker mode (job names as args): runs them sequentially on the current
# CUDA_VISIBLE_DEVICES.  Job names:
#   mhd_fno_van  mhd_fno_act  mhd_trans_van  mhd_trans_act  mhd_flower  sw_adrnet
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$ROOT/data/MHD_64_ckpts"
EXP="$ROOT/experiments"
DATA_ROOT="${DATA_ROOT:-/path/to/well_data}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/mhd64_eval}"
GPUS="${GPUS:-4,5,6,7}"
ROLLOUT_BS="${ROLLOUT_BS:-2}"   # MHD 64^3 is memory-heavy; keep small
mkdir -p "$OUT_DIR"

# Common flags shared by all MHD jobs
MHD_COMMON=(
  benchmark/eval_checkpoint.py
  --dataset MHD_64
  --data-root "$DATA_ROOT"
  --n-steps-input 4
  --batch-size 2
  --data-workers 4
  --rollout-batch-size "$ROLLOUT_BS"
  --spectral-gradient-3d
  --domain-length 1.0
)

job_args() {
  case "$1" in
    mhd_fno_van)
      echo "--ckpt $CKPT_DIR/fno_mhd64_ckpt.pt \
        --model ACTFNO3D --no-act --modes 32 --hidden 64"
      ;;
    mhd_fno_act)
      # Legacy deformers.* naming → remap
      echo "--ckpt $CKPT_DIR/act_fno_mhd64_ckpt.pt \
        --model ACTFNO3D --use-act --modes 32 --hidden 64 --max-disp 0.5 --remap-deformers"
      ;;
    mhd_trans_van)
      echo "--ckpt $CKPT_DIR/transolver_mhd64_ckpt.pt \
        --model TransolverPlus --no-act \
        --n-hidden 64 --n-head 8 --slice-num 32"
      ;;
    mhd_trans_act)
      # blocks.*.deform.* naming already up-to-date (no remap needed)
      echo "--ckpt $CKPT_DIR/act_transolver_mhd64_ckpt.pt \
        --model TransolverPlus --use-act --max-disp 0.5 \
        --n-hidden 64 --n-head 8 --slice-num 32 --num-deform-heads 16 --deform-hidden-dim 16"
      ;;
    mhd_unet_van)
      echo "--ckpt $CKPT_DIR/unet_mhd64_ckpt.pt \
        --model UNetConvNext --no-act \
        --blocks-per-stage 2 --init-features 42"
      ;;
    mhd_unet_act)
      echo "--ckpt $CKPT_DIR/act_unet_mhd64_ckpt.pt \
        --model ACTUNetConvNext --use-act --max-disp-null \
        --blocks-per-stage 2 --init-features 42"
      ;;
    mhd_flower)
      echo "--ckpt $EXP/MHD_64-flower_MHD_64-Flower-0.0005/20260726_164138/checkpoints/best.pt \
        --model Flower \
        --flower-lifting-dim 96 --flower-n-levels 4 --flower-num-heads 32 --flower-groups 32 \
        --boundary-condition-types PERIODIC"
      ;;
    sw_adrnet)
      # ADRNet is 2-D; eval on shallow_water test set
      echo "SPECIAL_SW_ADRNET"
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
    echo "unknown job: $name" >&2; return 1
  fi
  local log="$OUT_DIR/${name}.log"

  if [[ "$extra" == "SPECIAL_SW_ADRNET" ]]; then
    echo "[$(date +%H:%M:%S)] start $name (GPU=${CUDA_VISIBLE_DEVICES:-all}) -> $log"
    ( cd "$ROOT" && CONDA_NO_PLUGINS=true CUDA_DEVICE_ORDER=PCI_BUS_ID \
        conda run --no-capture-output -n thewell python benchmark/eval_checkpoint.py \
          --ckpt "$EXP/shallow_water-adrnet_shallow_water-ADRNet-0.0005/20260727_043347/checkpoints/best.pt" \
          --dataset shallow_water \
          --data-root "$DATA_ROOT" \
          --n-steps-input 4 \
          --batch-size 32 \
          --data-workers 4 \
          --rollout-batch-size 8 \
          --model ADRNet \
          --no-act \
          --adrnet-hidden 32 \
          --adrnet-nlayers 16 \
          --spectral-gradient \
          --domain-length 5.0 \
    ) > "$log" 2>&1
    echo "[$(date +%H:%M:%S)] done  $name (exit $?)"
    return
  fi

  echo "[$(date +%H:%M:%S)] start $name (GPU=${CUDA_VISIBLE_DEVICES:-all}) -> $log"
  ( cd "$ROOT" && CONDA_NO_PLUGINS=true CUDA_DEVICE_ORDER=PCI_BUS_ID \
      conda run --no-capture-output -n thewell python "${MHD_COMMON[@]}" $extra ) \
    > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $name (exit $?)"
}

if [[ $# -gt 0 ]]; then
  for job in "$@"; do run_one "$job"; done
  exit 0
fi

# Dispatch: 8 tasks × available GPUs
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
STREAMS=(
  "mhd_trans_act"
  "mhd_trans_van"
  "mhd_fno_act"
  "mhd_fno_van"
  "mhd_unet_act"
  "mhd_unet_van"
  "mhd_flower"
  "sw_adrnet"
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
