"""Measure forward FLOPs + param counts for the six Poisson-Gauss models.

Three architectures x {vanilla, ACT}:
  ACTFNO (use_act False/True), TransolverPlus (use_act False/True),
  UNetConvNext / ACTUNetConvNext.

Uses the same hyperparameters as ``benchmark/configs/model/*.yaml`` and the same
dim inference as ``eval_checkpoint.build_model``. FLOPs are counted with
``torch.utils.flop_counter.FlopCounterMode`` over a single-sample forward pass
(batch_size=1) at the Poisson-Gauss resolution (128x128).

Run from ACT-NeuralOP/ inside the ``thewell`` conda env:
    python benchmark/measure_flops.py
"""

import os.path as osp
import sys

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.flop_counter import FlopCounterMode

import benchmark.models as models

# Poisson-Gauss: 128x128 grid, 1 field, n_steps_input=1, no constant fields.
N_SPATIAL_DIMS = 2
SPATIAL_RESOLUTION = (128, 128)
N_FIELDS = 1
N_CONSTANT_FIELDS = 0
N_STEPS_INPUT = 1
DIM_IN = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS  # 1
DIM_OUT = N_FIELDS  # 1


def base_kwargs():
    return dict(
        dim_in=DIM_IN,
        dim_out=DIM_OUT,
        n_spatial_dims=N_SPATIAL_DIMS,
        spatial_resolution=SPATIAL_RESOLUTION,
        gradient_checkpointing=False,
    )


def build_models():
    """Return {label: nn.Module} for the six Poisson models, config-matched."""
    m = {}

    # --- FNO: act_fno.yaml (max_disp=0.5) ---
    fno_kwargs = dict(modes1=32, modes2=32, hidden_channels=64,
                      disp_net_branched=True, max_disp=0.5)
    m["vanilla_fno"] = models.ACTFNO(**base_kwargs(), use_act=False, **fno_kwargs)
    m["act_fno"] = models.ACTFNO(**base_kwargs(), use_act=True, **fno_kwargs)

    # --- Transolver: transolver_plus.yaml ---
    trans_common = dict(
        n_layers=4, n_hidden=64, dropout=0.0, n_head=8, mlp_ratio=4.0,
        slice_num=32, ref=8, unified_pos=False,
        num_deform_heads=16, deform_disp_net_branched=True, deform_hidden_dim=16,
        projection_channels=128,
    )
    m["vanilla_transolver"] = models.TransolverPlus(
        **base_kwargs(), use_act=False, deform_max_disp=None, **trans_common)
    # act_transolver_poisson.yaml: use_act=true, deform_max_disp=0.5
    m["act_transolver"] = models.TransolverPlus(
        **base_kwargs(), use_act=True, deform_max_disp=0.5, **trans_common)

    # --- UNet ConvNext: unet_convnext.yaml / act_unet_convnext.yaml ---
    m["vanilla_convnext"] = models.UNetConvNext(
        **base_kwargs(), blocks_per_stage=2, init_features=42)
    m["act_convnext"] = models.ACTUNetConvNext(
        **base_kwargs(), blocks_per_stage=2, init_features=42,
        num_heads=16, disp_net_branched=True, max_disp=None, deform_hidden_dim=16)

    return m


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure(model, device):
    model = model.to(device).eval()
    x = torch.randn(1, DIM_IN, *SPATIAL_RESOLUTION, device=device)
    flop_counter = FlopCounterMode(display=False)
    with torch.inference_mode(), flop_counter:
        out = model(x)
        # ACT models return (out, disps); ignore the aux output.
        if isinstance(out, (tuple, list)):
            out = out[0]
    total_flops = flop_counter.get_total_flops()
    return total_flops, tuple(out.shape)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  input=(1,{DIM_IN},{SPATIAL_RESOLUTION[0]},{SPATIAL_RESOLUTION[1]})\n")

    order = [
        "vanilla_fno", "act_fno",
        "vanilla_transolver", "act_transolver",
        "vanilla_convnext", "act_convnext",
    ]
    built = build_models()

    rows = []
    for name in order:
        model = built[name]
        params = count_params(model)
        try:
            flops, out_shape = measure(model, device)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            rows.append((name, params, None))
            continue
        rows.append((name, params, flops))
        print(f"{name:20s}  params={params/1e6:8.3f}M  "
              f"FLOPs={flops/1e9:10.3f} GFLOPs  out={out_shape}")

    print("\n==== SUMMARY (batch=1, fwd) ====")
    print(f"{'model':20s} {'params (M)':>12s} {'GFLOPs':>12s}")
    for name, params, flops in rows:
        f = f"{flops/1e9:12.3f}" if flops is not None else f"{'ERR':>12s}"
        print(f"{name:20s} {params/1e6:12.3f} {f}")


if __name__ == "__main__":
    main()
