"""Unified checkpoint evaluator for the ACT-NeuralOP benchmark.

Rebuilds a model, loads a checkpoint, and runs the trainer's ``validation_loop`` on a
dataset's test split — single-step and (optionally) full-trajectory rollout — reporting
MSE / NRMSE / VRMSE in physical units, using the exact same pipeline as training so
numbers are directly comparable across models.

Works for any registered Well dataset (ns, darcy, shallow_water, ...) and any model in
``benchmark.models``. Handles the common gotchas we hit:

* ``--max-disp``: ACT-FNO deform scale. Checkpoints trained with the legacy
  ``MultiHeadDeformBlock`` used ``max_disp=0.5``; the current ``ACTFNO`` default (``None``)
  is a one-cell bound (~0.0157 at 128x128) that mis-scales a loaded deform. Default here is
  0.5. Irrelevant when ``--no-act``.
* ``--remap-deformers``: legacy ACT-FNO checkpoints name deform blocks ``model.deformers.*``;
  current code uses ``model.act_blocks.*``. Enable to rename on load.
* ``--no-rollout``: steady-state datasets (e.g. darcy, 1->1) have no meaningful rollout.
* ``--ns-physical``: NS vorticity spectral/gradient/conservation (periodic torus).
* ``--spectral-gradient``: general scalar-field spectral/gradient metrics (e.g. shallow
  water height). Averages over channels when C>1.

Examples (from ACT-NeuralOP/):
    # ns pretrained ACT-FNO (legacy naming, needs remap)
    python benchmark/eval_checkpoint.py --ckpt ../ns_incom_128_T50_ckpts/act_fno_ckpt.pt \\
        --dataset ns --n-steps-input 4 --model ACTFNO --use-act --max-disp 0.5 --remap-deformers

    # shallow water paper FNO+ACT
    python benchmark/eval_checkpoint.py \\
        --ckpt /path/to/shallow_water-deformer_fno-.../best.pt \\
        --dataset shallow_water --data-root /path/to/well_data --n-steps-input 4 \\
        --model ACTFNO --use-act --max-disp 0.5 --remap-deformers \\
        --spectral-gradient --domain-length 5.0
"""

import argparse
import os.path as osp
import sys
from collections import defaultdict

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

import benchmark.models as models
from benchmark.metrics import MSE
from benchmark.trainer import Trainer
from data import WellDataModule
from data.normalization import ZScoreNormalization


def build_datamodule(args) -> WellDataModule:
    base = args.data_root if osp.isabs(args.data_root) else osp.join(PROJECT_ROOT, args.data_root)
    return WellDataModule(
        well_base_path=base,
        well_dataset_name=args.dataset,
        batch_size=args.batch_size,
        n_steps_input=args.n_steps_input,
        n_steps_output=1,
        use_normalization=True,
        normalization_type=ZScoreNormalization,
        min_dt_stride=1,
        max_dt_stride=1,
        data_workers=args.data_workers,
    )


def rollout_test_loader(dm: WellDataModule, batch_size: int, data_workers: int):
    """Rollout test loader with a tunable batch size.

    ``WellDataModule.rollout_test_dataloader`` hardcodes ``batch_size=1``, which leaves the
    GPU mostly idle across a ~100-step autoregressive rollout. Trajectories within a split
    have equal length, so they batch cleanly. ``drop_last=False`` keeps every trajectory.
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dm.rollout_test_dataset,
        num_workers=data_workers,
        pin_memory=True,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def load_state(ckpt_path: str, remap_deformers: bool):
    """Load ``model_state_dict``; optionally rename legacy ``deformers`` -> ``act_blocks``."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    if remap_deformers:
        sd = {k.replace("model.deformers.", "model.act_blocks."): v for k, v in sd.items()}
    return sd, ckpt.get("epoch"), ckpt.get("validation_loss")


def build_model(args, meta):
    """Instantiate the requested model class with dims inferred from dataset metadata."""
    n_in = args.n_steps_input * meta.n_fields + meta.n_constant_fields
    model_cls = getattr(models, args.model)
    kwargs = dict(
        dim_in=n_in,
        dim_out=meta.n_fields,
        n_spatial_dims=meta.n_spatial_dims,
        spatial_resolution=meta.spatial_resolution,
        gradient_checkpointing=False,
    )
    max_disp = args.max_disp if args.max_disp_is_set else None
    if args.model == "ACTFNO":
        kwargs.update(
            modes1=args.modes,
            modes2=args.modes,
            hidden_channels=args.hidden,
            use_act=args.use_act,
            max_disp=max_disp,
        )
    elif args.model in ("UNetConvNext", "UNet_ConvNext"):
        kwargs.update(
            blocks_per_stage=args.blocks_per_stage,
            init_features=args.init_features,
        )
    elif args.model == "ACTUNetConvNext":
        kwargs.update(
            blocks_per_stage=args.blocks_per_stage,
            init_features=args.init_features,
            num_heads=args.num_deform_heads,
            disp_net_branched=True,
            max_disp=max_disp,
            deform_hidden_dim=args.deform_hidden_dim,
        )
    elif args.model == "TransolverPlus":
        kwargs.update(
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            n_head=args.n_head,
            slice_num=args.slice_num,
            use_act=args.use_act,
            num_deform_heads=args.num_deform_heads,
            deform_disp_net_branched=True,
            deform_max_disp=max_disp if args.use_act else None,
            deform_hidden_dim=args.deform_hidden_dim,
            projection_channels=args.projection_channels,
        )
    elif args.model == "ACTFNO3D":
        modes3 = args.modes3 if args.modes3 is not None else args.modes
        kwargs.update(
            modes1=args.modes,
            modes2=args.modes,
            modes3=modes3,
            hidden_channels=args.hidden,
            use_act=args.use_act,
            max_disp=max_disp,
        )
    elif args.model == "Flower":
        kwargs.update(
            lifting_dim=args.flower_lifting_dim,
            n_levels=args.flower_n_levels,
            num_heads=args.flower_num_heads,
            groups=args.flower_groups,
            boundary_condition_types=args.boundary_condition_types,
        )
    elif args.model == "ADRNet":
        kwargs.update(
            hidden_channels=args.adrnet_hidden,
            nlayers=args.adrnet_nlayers,
        )
    else:
        # Best-effort: only pass ACTFNO-like knobs when the signature accepts them.
        pass
    return model_cls(**kwargs)


def grab(d, needle):
    """First value whose key holds ``needle``, aggregated over all time (``full``, ``T=all``)."""
    for k, v in d.items():
        if needle in k and "T=all" in k and "full" in k:
            return v
    return None


def _print_rollout_horizons(per_step_nrmse: list[float], horizons: list[int]) -> None:
    """Print NRMSE at selected lead times and mean over the first H steps."""
    T = len(per_step_nrmse)
    if T == 0:
        return
    print(f"\n==== ROLLOUT NRMSE BY HORIZON (T={T} steps) ====")
    print(f"  {'H':>4}  {'NRMSE@H':>12}  {'mean[1..H]':>12}")
    seen = set()
    for h in horizons:
        if h < 1 or h > T or h in seen:
            continue
        seen.add(h)
        at_h = per_step_nrmse[h - 1]
        prefix = sum(per_step_nrmse[:h]) / h
        print(f"  {h:4d}  {at_h:12.6f}  {prefix:12.6f}")
    if T not in seen:
        at_h = per_step_nrmse[T - 1]
        prefix = sum(per_step_nrmse) / T
        print(f"  {T:4d}  {at_h:12.6f}  {prefix:12.6f}")


def _print_mass_horizons(
    mass_rel_err: list[float], mass_ratio: list[float], horizons: list[int]
) -> None:
    """Print density-mass error at selected autoregressive lead times."""
    T = len(mass_rel_err)
    if T == 0:
        return
    print(f"\n==== DENSITY MASS CONSERVATION BY HORIZON (T={T} steps) ====")
    print(f"  {'H':>4}  {'|Mpred-Mtrue|/|Mtrue|':>23}  {'Mpred/Mtrue':>13}")
    seen = set()
    for h in [*horizons, T]:
        if h < 1 or h > T or h in seen:
            continue
        seen.add(h)
        print(f"  {h:4d}  {mass_rel_err[h - 1]:23.6e}  {mass_ratio[h - 1]:13.6f}")


def _average_rollout_field_metrics(
    trainer,
    dataloader,
    report_fn,
    domain_length: float,
    meta,
    mass_channel: int | None = None,
):
    """One rollout pass: NRMSE/MSE (physical) + optional ``report_fn`` over snapshots.

    ``y_pred`` / ``y_ref`` are ``(B, T, ..., C)`` in physical units. Each channel is scored
    separately for spectral/gradient and then averaged (SW height has C=1).

    Also accumulates per-step NRMSE (mean over batch & channels) for short-horizon curves.
    """
    from benchmark.metrics.spatial import MSE as MSEMetric
    from benchmark.metrics.spatial import NRMSE, VRMSE

    sums: dict[str, float] = defaultdict(float)
    count = 0
    mse_chunks, nrmse_chunks, vrmse_chunks = [], [], []
    step_nrmse_sum = None
    step_nrmse_count = None
    mass_rel_sum = None
    mass_ratio_sum = None
    mass_count = None
    trainer.model.eval()
    with torch.inference_mode():
        for batch in dataloader:
            y_pred, y_ref = trainer.rollout_model(
                trainer.model, batch, trainer.formatter, train=False
            )
            # Metric.eval averages over spatial dims -> (B, T, C); mean over all for T=all.
            mse_btc = MSEMetric.eval(y_pred, y_ref, meta)
            nrmse_btc = NRMSE.eval(y_pred, y_ref, meta)
            vrmse_btc = VRMSE.eval(y_pred, y_ref, meta)
            mse_chunks.append(mse_btc.mean().item())
            nrmse_chunks.append(nrmse_btc.mean().item())
            vrmse_chunks.append(vrmse_btc.mean().item())

            # Per-step NRMSE: mean over batch and channels -> (T,)
            nrmse_t = nrmse_btc.detach().float().mean(dim=(0, -1)).cpu()
            bc = float(nrmse_btc.shape[0] * nrmse_btc.shape[-1])
            if step_nrmse_sum is None:
                step_nrmse_sum = torch.zeros(nrmse_t.numel())
                step_nrmse_count = torch.zeros(nrmse_t.numel())
            T = nrmse_t.numel()
            step_nrmse_sum[:T] += nrmse_t * bc
            step_nrmse_count[:T] += bc

            if mass_channel is not None:
                # MHD density is channel 0. Integrating via mean is equivalent to
                # sum*dV for a fixed grid and avoids a domain-volume scale factor.
                spatial_dims = tuple(range(2, y_pred.ndim - 1))
                mass_p = y_pred[..., mass_channel].mean(dim=spatial_dims)
                mass_t = y_ref[..., mass_channel].mean(dim=spatial_dims)
                denom = mass_t.abs().clamp_min(1e-12)
                rel_bt = (mass_p - mass_t).abs() / denom
                ratio_bt = mass_p / mass_t.clamp_min(1e-12)
                rel_t = rel_bt.detach().float().sum(dim=0).cpu()
                ratio_t = ratio_bt.detach().float().sum(dim=0).cpu()
                if mass_rel_sum is None:
                    mass_rel_sum = torch.zeros(rel_t.numel())
                    mass_ratio_sum = torch.zeros(ratio_t.numel())
                    mass_count = torch.zeros(rel_t.numel())
                Tmass = rel_t.numel()
                mass_rel_sum[:Tmass] += rel_t
                mass_ratio_sum[:Tmass] += ratio_t
                mass_count[:Tmass] += rel_bt.shape[0]

            if report_fn is not None:
                yp = y_pred.detach().float().cpu().numpy()
                yr = y_ref.detach().float().cpu().numpy()
                B, Tdim = yp.shape[0], yp.shape[1]
                C = yp.shape[-1]
                for b in range(B):
                    for t in range(Tdim):
                        for c in range(C):
                            rep = report_fn(yp[b, t, ..., c], yr[b, t, ..., c], L=domain_length)
                            for k, v in rep.items():
                                sums[k] += float(v)
                            count += 1
    phys = {k: v / max(count, 1) for k, v in sums.items()} if report_fn is not None else {}
    errors = {
        "MSE": float(sum(mse_chunks) / max(len(mse_chunks), 1)),
        "NRMSE": float(sum(nrmse_chunks) / max(len(nrmse_chunks), 1)),
        "VRMSE": float(sum(vrmse_chunks) / max(len(vrmse_chunks), 1)),
    }
    if step_nrmse_sum is None:
        per_step_nrmse: list[float] = []
    else:
        per_step_nrmse = (step_nrmse_sum / step_nrmse_count.clamp_min(1.0)).tolist()
    if mass_rel_sum is None:
        mass_curves = None
    else:
        mass_curves = {
            "rel_err": (mass_rel_sum / mass_count.clamp_min(1.0)).tolist(),
            "ratio": (mass_ratio_sum / mass_count.clamp_min(1.0)).tolist(),
        }
    return phys, errors, count, per_step_nrmse, mass_curves


def run_rollout_errors(trainer, dataloader, meta, mass_channel=None):
    """Rollout MSE/NRMSE/VRMSE + per-step NRMSE curve (no spectral extras)."""
    return _average_rollout_field_metrics(
        trainer, dataloader, None, 1.0, meta, mass_channel=mass_channel
    )


def run_ns_physical(trainer, dataloader, meta, domain_length=1.0, mass_channel=None):
    """Average NS physical/spectral/gradient metrics over a rollout dataloader."""
    from benchmark.metrics.ns_physical import ns_physical_report

    return _average_rollout_field_metrics(
        trainer, dataloader, ns_physical_report, domain_length, meta, mass_channel
    )


def run_spectral_gradient(trainer, dataloader, meta, domain_length=1.0, mass_channel=None):
    """Average scalar-field spectral/gradient metrics over a rollout dataloader."""
    from benchmark.metrics.ns_physical import scalar_field_report

    return _average_rollout_field_metrics(
        trainer, dataloader, scalar_field_report, domain_length, meta, mass_channel
    )


def run_spectral_gradient_3d(
    trainer, dataloader, meta, domain_length=1.0, mass_channel=None
):
    """Average 3-D scalar-field spectral/gradient metrics over a rollout dataloader."""
    from benchmark.metrics.ns_physical import scalar_field_report_3d

    return _average_rollout_field_metrics(
        trainer, dataloader, scalar_field_report_3d, domain_length, meta, mass_channel
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Path to a checkpoint .pt (ACT training format).")
    ap.add_argument("--dataset", required=True, help="Well dataset name, e.g. ns / darcy / shallow_water.")
    ap.add_argument("--data-root", default="data_root", help="Base dir holding <dataset>/.")
    ap.add_argument("--model", default="ACTFNO", help="Class name in benchmark.models.")
    ap.add_argument("--n-steps-input", type=int, default=4, help="Input frames (ns/SW=4, darcy=1).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--data-workers", type=int, default=6)
    # ACT / FNO knobs
    ap.add_argument("--use-act", dest="use_act", action="store_true")
    ap.add_argument("--no-act", dest="use_act", action="store_false")
    ap.set_defaults(use_act=True)
    ap.add_argument("--modes", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument(
        "--max-disp",
        type=float,
        default=0.5,
        help="ACT deform scale (ckpt-dependent). For UNet paper ckpts use with --max-disp-null.",
    )
    ap.add_argument(
        "--max-disp-null",
        dest="max_disp_is_set",
        action="store_false",
        help="Pass max_disp=None (one-cell bound). Needed for paper UNet+ACT ckpts.",
    )
    ap.set_defaults(max_disp_is_set=True)
    ap.add_argument("--remap-deformers", action="store_true", help="Rename legacy deformers.* keys.")
    # UNet / Transolver knobs (paper SW/DR defaults)
    ap.add_argument("--blocks-per-stage", type=int, default=2)
    ap.add_argument("--init-features", type=int, default=42)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-hidden", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--slice-num", type=int, default=32)
    ap.add_argument("--num-deform-heads", type=int, default=16)
    ap.add_argument("--deform-hidden-dim", type=int, default=16)
    ap.add_argument("--projection-channels", type=int, default=128)
    # ACTFNO3D knobs
    ap.add_argument("--modes3", type=int, default=None, help="3rd spectral mode count for ACTFNO3D (defaults to --modes).")
    # Flower knobs
    ap.add_argument("--flower-lifting-dim", type=int, default=96)
    ap.add_argument("--flower-n-levels", type=int, default=4)
    ap.add_argument("--flower-num-heads", type=int, default=32)
    ap.add_argument("--flower-groups", type=int, default=32)
    ap.add_argument("--boundary-condition-types", nargs="+", default=["PERIODIC"])
    # ADRNet knobs
    ap.add_argument("--adrnet-hidden", type=int, default=32)
    ap.add_argument("--adrnet-nlayers", type=int, default=16)
    # rollout
    ap.add_argument(
        "--no-rollout",
        dest="rollout",
        action="store_false",
        help="Skip rollout test (steady-state datasets).",
    )
    ap.set_defaults(rollout=True)
    ap.add_argument(
        "--skip-single-step",
        action="store_true",
        help="Skip single-step test (sliding-window) — much faster for long SW trajectories.",
    )
    ap.add_argument("--max-rollout-steps", type=int, default=200)
    ap.add_argument(
        "--rollout-batch-size",
        type=int,
        default=8,
        help="Trajectories per rollout batch (datamodule default is 1, which is GPU-starved).",
    )
    ap.add_argument(
        "--ns-physical",
        action="store_true",
        help="Also report NS spectral/gradient/conservation metrics on rollout "
        "(single-field vorticity on a periodic grid only).",
    )
    ap.add_argument(
        "--spectral-gradient",
        action="store_true",
        help="Also report scalar-field spectral/gradient metrics on rollout "
        "(e.g. shallow_water height; averages over channels if C>1).",
    )
    ap.add_argument(
        "--spectral-gradient-3d",
        action="store_true",
        help="Like --spectral-gradient but uses 3-D FFT / volumetric gradients "
        "(for 3-D datasets such as MHD_64; averages over channels if C>1).",
    )
    ap.add_argument(
        "--mhd-density-mass",
        action="store_true",
        help="Report per-horizon density mass error for MHD (density must be channel 0).",
    )
    ap.add_argument(
        "--domain-length",
        type=float,
        default=1.0,
        help="Physical side length L of the square domain (ns=1.0, shallow_water=5.0).",
    )
    ap.add_argument(
        "--rollout-horizons",
        type=str,
        default="1,2,4,8,16,32,48,64,80,96",
        help="Comma-separated lead times H for NRMSE@H and mean[1..H] (empty to skip).",
    )
    args = ap.parse_args()
    horizons = [int(x) for x in args.rollout_horizons.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = build_datamodule(args)
    meta = dm.train_dataset_core.metadata

    max_disp_for_log = args.max_disp if args.max_disp_is_set else None
    model = build_model(args, meta)
    state, epoch, ck_val = load_state(args.ckpt, args.remap_deformers)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.ckpt}")
    print(
        f"  dataset={args.dataset} model={args.model} n_steps_input={args.n_steps_input} "
        f"use_act={args.use_act} max_disp={max_disp_for_log} remap={args.remap_deformers}"
    )
    print(f"  ckpt epoch={epoch} ckpt.validation_loss={ck_val}")
    print(f"  load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  MISSING:", missing[:6])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:6])
    model = model.to(device).eval()

    trainer = Trainer(
        checkpoint_folder="/tmp/eval_ckpt",
        artifact_folder="/tmp/eval_ckpt",
        viz_folder="/tmp/eval_ckpt/viz",
        formatter="channels_first_default",
        model=model,
        datamodule=dm,
        optimizer=None,
        loss_fn=MSE(),
        epochs=1,
        checkpoint_frequency=0,
        val_frequency=1,
        rollout_val_frequency=1,
        max_rollout_steps=args.max_rollout_steps,
        short_validation_length=10_000,
        make_rollout_videos=False,
        num_time_intervals=4,
        device=device,
        is_distributed=False,
    )

    print("\n==== RESULTS (physical units) ====")
    if not args.skip_single_step:
        test_loss, test_dict = trainer.validation_loop(
            dm.test_dataloader(), valid_or_test="test", full=True, epoch=0
        )
        print(f"  test  MSE (single-step)   = {test_loss:.6e}")
        print(f"  test  NRMSE (single-step) = {grab(test_dict, 'NRMSE')}")
        print(f"  test  VRMSE (single-step) = {grab(test_dict, 'VRMSE')}")
    else:
        print("  (single-step skipped)")

    # Prefer a single fused rollout pass when spectral/physical metrics are requested,
    # so we do not autoregress the test set twice. Always use the custom loop when
    # horizon curves are requested (trainer.validation_loop does not expose per-step).
    want_spectral = args.ns_physical or args.spectral_gradient or args.spectral_gradient_3d
    fused = want_spectral and args.rollout
    use_custom_rollout = args.rollout and (
        fused or bool(horizons) or args.mhd_density_mass
    )
    if use_custom_rollout:
        if sum([args.ns_physical, args.spectral_gradient, args.spectral_gradient_3d]) > 1:
            raise SystemExit("Use only one of --ns-physical / --spectral-gradient / --spectral-gradient-3d per run.")
        loader = rollout_test_loader(dm, args.rollout_batch_size, args.data_workers)
        mass_channel = 0 if args.mhd_density_mass else None
        if args.ns_physical:
            phys, errors, n, per_step, mass_curves = run_ns_physical(
                trainer,
                loader,
                meta,
                domain_length=args.domain_length,
                mass_channel=mass_channel,
            )
            title = "NS PHYSICAL / SPECTRAL / GRADIENT"
        elif args.spectral_gradient_3d:
            phys, errors, n, per_step, mass_curves = run_spectral_gradient_3d(
                trainer,
                loader,
                meta,
                domain_length=args.domain_length,
                mass_channel=mass_channel,
            )
            title = "SPECTRAL / GRADIENT (3-D scalar field)"
        elif args.spectral_gradient:
            phys, errors, n, per_step, mass_curves = run_spectral_gradient(
                trainer,
                loader,
                meta,
                domain_length=args.domain_length,
                mass_channel=mass_channel,
            )
            title = "SPECTRAL / GRADIENT (scalar field)"
        else:
            phys, errors, n, per_step, mass_curves = run_rollout_errors(
                trainer, loader, meta, mass_channel=mass_channel
            )
            title = None
        print(f"  rollout test MSE   = {errors['MSE']:.6e}")
        print(f"  rollout test NRMSE = {errors['NRMSE']}")
        print(f"  rollout test VRMSE = {errors['VRMSE']}")
        print(f"  rollout steps      = {len(per_step)}")
        if horizons:
            _print_rollout_horizons(per_step, horizons)
        if mass_curves is not None:
            _print_mass_horizons(
                mass_curves["rel_err"], mass_curves["ratio"], horizons
            )
        if title is not None:
            print(f"\n==== {title} (rollout, {n} snapshots) ====")
            print("  [gradient — ratios <1 mean over-smoothing]")
            for k in ("grad_rel_err", "grad_max_ratio", "grad_p99_ratio", "TV_ratio"):
                print(f"    {k:22s}= {phys[k]:.4f}")
            print("  [spectrum — band relative error; high band = spectral bias]")
            for k in ("specerr_low", "specerr_mid", "specerr_high", "spectral_centroid_ratio"):
                print(f"    {k:22s}= {phys[k]:.4f}")
            if args.ns_physical:
                print("  [conservation — relative errors vs ground truth]")
                for k in ("mass_abs_err", "momentum_err", "div_rms", "energy_err", "enstrophy_err"):
                    print(f"    {k:22s}= {phys[k]:.4f}")
            else:
                print("  [scalar mass / energy vs ground truth]")
                for k in ("mass_rel_err", "energy_err"):
                    print(f"    {k:22s}= {phys[k]:.4f}")
    elif args.rollout:
        roll_loss, roll_dict = trainer.validation_loop(
            rollout_test_loader(dm, args.rollout_batch_size, args.data_workers),
            valid_or_test="rollout_test",
            full=True,
            epoch=0,
        )
        print(f"  rollout test MSE   = {roll_loss:.6e}")
        print(f"  rollout test NRMSE = {grab(roll_dict, 'NRMSE')}")
        print(f"  rollout test VRMSE = {grab(roll_dict, 'VRMSE')}")


if __name__ == "__main__":
    main()
