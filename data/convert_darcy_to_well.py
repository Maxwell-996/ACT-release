#!/usr/bin/env python
"""Convert the PDEBench ``2D_DarcyFlow`` HDF5 into The Well HDF5 format.

Source
------
``2D_DarcyFlow_beta1.0_Train.hdf5`` (PDEBench) contains:
    nu            : (N, 128, 128) float32  -- diffusion coefficient a(x)  (the INPUT)
    tensor        : (N, 1, 128, 128) float32 -- steady-state solution u(x) (the TARGET)
    x-coordinate  : (128,)
    y-coordinate  : (128,)

DarcyFlow is a **steady-state** (time-independent) problem: learn the operator
``a(x) -> u(x)``. This benchmark framework is temporal (predicts the next step of a
field), so we encode each sample as a **2-step "trajectory"**:

    t = 0 : darcy   := nu   (input  field, fed to the model)
    t = 1 : darcy   := u    (target field, the solution)

with ``n_steps_input=1, n_steps_output=1`` there is exactly one input->output window
per sample and the model learns ``nu -> u`` directly (absolute, not delta).

Processing
----------
1. Split **by sample** 8:1:1 -> train / valid / test (deterministic prefix ordering).
2. Write one Well HDF5 per split with a single scalar field ``darcy`` of shape
   ``(n_samples, 2, 128, 128)`` (sample- and time-varying), a cartesian x/y grid,
   a 2-entry time axis, and ``wall`` (Dirichlet) boundaries.
3. Emit ``stats.yaml`` from the **train** split. Because input (nu) and target (u)
   share the single field's mean/std, stats are computed over the **combined**
   nu+u train distribution.

Layout produced under ``--out-dir``::

    <out-dir>/darcy/
        stats.yaml
        data/train/darcy.hdf5
        data/valid/darcy.hdf5
        data/test/darcy.hdf5

matching ``WellDataset(well_base_path=<out-dir>, well_dataset_name="darcy")``.
"""

import argparse
import os

import h5py
import numpy as np

DATASET_NAME = "darcy"
FIELD_NAME = "darcy"
BOUNDARY_TYPE = "wall"  # DarcyFlow uses Dirichlet boundary conditions.


def _write_split_hdf5(out_file: str, data: np.ndarray) -> None:
    """Write one split to a Well-format HDF5 file.

    Args:
        out_file: Destination ``.hdf5`` path.
        data: ``(n_samples, 2, nx, ny)`` float32 array. Axis 1 is the pseudo-time
            axis: index 0 = nu (input), index 1 = u (target).
    """
    n_traj, n_time, nx, ny = data.shape
    assert n_time == 2, f"Expected 2 pseudo-time steps (nu, u), got {n_time}"
    x_coord = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_coord = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    t_coord = np.linspace(0.0, 1.0, n_time, dtype=np.float32)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with h5py.File(out_file, "w") as dst:
        dst.attrs["grid_type"] = "cartesian"
        dst.attrs["n_spatial_dims"] = 2
        dst.attrs["dataset_name"] = DATASET_NAME
        dst.attrs["n_trajectories"] = n_traj

        dims = dst.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        x_ds = dims.create_dataset("x", data=x_coord)
        x_ds.attrs["sample_varying"] = False
        x_ds.attrs["time_varying"] = False
        y_ds = dims.create_dataset("y", data=y_coord)
        y_ds.attrs["sample_varying"] = False
        y_ds.attrs["time_varying"] = False
        time_ds = dims.create_dataset("time", data=t_coord)
        time_ds.attrs["sample_varying"] = False
        time_ds.attrs["time_varying"] = False

        # Single scalar field (tensor order 0), sample- and time-varying.
        t0 = dst.create_group("t0_fields")
        t0.attrs["field_names"] = [FIELD_NAME]
        field_ds = t0.create_dataset(FIELD_NAME, data=data.astype(np.float32))
        field_ds.attrs["dim_varying"] = np.array([True, True])
        field_ds.attrs["sample_varying"] = True
        field_ds.attrs["time_varying"] = True
        field_ds.attrs["symmetric"] = False

        # No higher-order fields, no scalars.
        dst.create_group("t1_fields").attrs["field_names"] = np.array([], dtype="S")
        dst.create_group("t2_fields").attrs["field_names"] = np.array([], dtype="S")
        dst.create_group("scalars").attrs["field_names"] = np.array([], dtype="S")

        bc = dst.create_group("boundary_conditions")
        bc_x = bc.create_group("x_boundary")
        bc_x.attrs["bc_type"] = BOUNDARY_TYPE
        bc_x.attrs["associated_dims"] = ["x"]
        bc_x.create_dataset("mask", data=np.ones((nx, ny), dtype=bool))
        bc_y = bc.create_group("y_boundary")
        bc_y.attrs["bc_type"] = BOUNDARY_TYPE
        bc_y.attrs["associated_dims"] = ["y"]
        bc_y.create_dataset("mask", data=np.ones((nx, ny), dtype=bool))

    print(f"  wrote {out_file}: {n_traj} samples x {n_time} steps x {nx}x{ny}")


def _write_stats_yaml(out_path: str, train: np.ndarray) -> None:
    """Compute normalization statistics from the train split and write ``stats.yaml``.

    ``ZScoreNormalization`` requires the keys ``mean``, ``std``, ``mean_delta``,
    ``std_delta``; we also emit ``rms``/``rms_delta`` for RMS-normalization
    compatibility. Because the single ``darcy`` field carries both the input (nu,
    t=0) and the target (u, t=1), stats are computed over the **combined** nu+u
    train distribution so normalization is well-behaved for both. Deltas are the
    (u - nu) differences across the pseudo-time axis.
    """
    flat = train.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std())
    rms = float(np.sqrt((flat**2).mean()))

    delta = (train[:, 1:] - train[:, :-1]).reshape(-1)
    mean_delta = float(delta.mean())
    std_delta = float(delta.std())
    rms_delta = float(np.sqrt((delta**2).mean()))

    lines = [
        "mean:",
        f"  {FIELD_NAME}: {mean:.8E}",
        "std:",
        f"  {FIELD_NAME}: {std:.8E}",
        "rms:",
        f"  {FIELD_NAME}: {rms:.8E}",
        "mean_delta:",
        f"  {FIELD_NAME}: {mean_delta:.8E}",
        "std_delta:",
        f"  {FIELD_NAME}: {std_delta:.8E}",
        "rms_delta:",
        f"  {FIELD_NAME}: {rms_delta:.8E}",
        "",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(
        f"  wrote {out_path}: mean={mean:.4e} std={std:.4e} "
        f"mean_delta={mean_delta:.4e} std_delta={std_delta:.4e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDEBench 2D_DarcyFlow HDF5 to The Well HDF5"
    )
    parser.add_argument(
        "--src",
        default="/path/to/DarcyFlow/2D_DarcyFlow_beta1.0_Train.hdf5",
        help="Path to the PDEBench 2D_DarcyFlow HDF5 file.",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_root"
        ),
        help="Base output dir; dataset is written under <out-dir>/darcy.",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions (default 0.8 0.1 0.1).",
    )
    args = parser.parse_args()

    print(f"Loading {args.src}")
    with h5py.File(args.src, "r") as f:
        nu = f["nu"][:]  # (N, 128, 128)
        u = f["tensor"][:, 0]  # (N, 128, 128) -- drop singleton channel axis
    print(f"  nu {nu.shape}, u {u.shape}")
    assert nu.shape == u.shape, f"nu/u shape mismatch: {nu.shape} vs {u.shape}"

    # Stack into a 2-step pseudo-trajectory: [t0=nu, t1=u] -> (N, 2, nx, ny).
    data = np.stack([nu, u], axis=1).astype(np.float32)
    print(f"  stacked -> {data.shape} (axis1: [nu, u])")

    n_total = data.shape[0]
    f_tr, f_va, f_te = args.split
    assert abs(f_tr + f_va + f_te - 1.0) < 1e-6, "Split fractions must sum to 1."
    n_train = int(round(n_total * f_tr))
    n_val = int(round(n_total * f_va))
    n_test = n_total - n_train - n_val

    train = data[:n_train]
    valid = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    print(
        f"  split by sample -> train {train.shape[0]}, "
        f"valid {valid.shape[0]}, test {test.shape[0]}"
    )

    ds_root = os.path.join(args.out_dir, DATASET_NAME)
    _write_split_hdf5(os.path.join(ds_root, "data", "train", "darcy.hdf5"), train)
    _write_split_hdf5(os.path.join(ds_root, "data", "valid", "darcy.hdf5"), valid)
    _write_split_hdf5(os.path.join(ds_root, "data", "test", "darcy.hdf5"), test)
    _write_stats_yaml(os.path.join(ds_root, "stats.yaml"), train)

    print(f"Done. Dataset root: {ds_root}")


if __name__ == "__main__":
    main()
