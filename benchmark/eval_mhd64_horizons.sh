#!/usr/bin/env bash
# MHD_64 rollout NRMSE horizon curves only (no spectral / no single-step).
# Faster than full eval_mhd64_ckpts.sh; reports NRMSE@H and mean[1..H].
#
#   GPUS=0,1,2,3,4,5,6 bash benchmark/eval_mhd64_horizons.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="$ROOT/data/MHD_64_ckpts"
EXP="$ROOT/experiments"
DATA_ROOT="${DATA_ROOT:-/path/to/well_data}"
OUT_DIR="${OUT_DIR:-$ROOT/experiments/mhd64_eval_horizons}"
GPUS="${GPUS:-0,1,2,3,4,5,6}"
ROLLOUT_BS="${ROLLOUT_BS:-2}"
HORIZONS="${HORIZONS:-1,2,4,8,16,24,32,48,64,80,96}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-200}"
# Set SPECTRAL_3D=1 to also report 3-D spectral/gradient metrics over the truncated rollout.
SPECTRAL_3D="${SPECTRAL_3D:-0}"
# Set MHD_DENSITY_MASS=0 to skip density-mass horizon report (speeds up spectral-only runs slightly).
MHD_DENSITY_MASS="${MHD_DENSITY_MASS:-1}"
mkdir -p "$OUT_DIR"

MHD_COMMON=(
  benchmark/eval_checkpoint.py
  --dataset MHD_64
  --data-root "$DATA_ROOT"
  --n-steps-input 4
  --batch-size 2
  --data-workers 4
  --rollout-batch-size "$ROLLOUT_BS"
  --max-rollout-steps "$MAX_ROLLOUT_STEPS"
  --skip-single-step
  --rollout-horizons "$HORIZONS"
  --domain-length 1.0
)
if [[ "$MHD_DENSITY_MASS" == "1" ]]; then
  MHD_COMMON+=(--mhd-density-mass)
fi
if [[ "$SPECTRAL_3D" == "1" ]]; then
  MHD_COMMON+=(--spectral-gradient-3d)
fi

job_args() {
  case "$1" in
    mhd_fno_van)
      echo "--ckpt $CKPT_DIR/fno_mhd64_ckpt.pt \
        --model ACTFNO3D --no-act --modes 32 --hidden 64"
      ;;
    mhd_fno_act)
      echo "--ckpt $CKPT_DIR/act_fno_mhd64_ckpt.pt \
        --model ACTFNO3D --use-act --modes 32 --hidden 64 --max-disp 0.5 --remap-deformers"
      ;;
    mhd_trans_van)
      echo "--ckpt $CKPT_DIR/transolver_mhd64_ckpt.pt \
        --model TransolverPlus --no-act \
        --n-hidden 64 --n-head 8 --slice-num 32"
      ;;
    mhd_trans_act)
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

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
STREAMS=(
  mhd_trans_act
  mhd_trans_van
  mhd_fno_act
  mhd_fno_van
  mhd_unet_act
  mhd_unet_van
  mhd_flower
)

pids=()
for i in "${!STREAMS[@]}"; do
  gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
  echo "stream $i on GPU $gpu: ${STREAMS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" bash "$0" "${STREAMS[$i]}" &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "All streams finished (fail=$fail). Logs in $OUT_DIR"
exit "$fail"
