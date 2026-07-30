#!/usr/bin/env python
"""Convert the Poisson-Gauss NetCDF4 dataset to The Well HDF5 format.

Source
------
``Poisson-Gauss.nc`` contains:
    source   : (20000, 128, 128) float32  -- Gaussian source term f(x)  (INPUT)
    solution : (20000, 128, 128) float32  -- Poisson solution u(x)       (TARGET)

where -Δu = f with **zero Dirichlet boundary conditions** (u=0 on boundary).

This is a steady-state problem identical in structure to DarcyFlow. Each sample
is encoded as a 2-step pseudo-trajectory:

    t = 0 : poisson := source   (input  field)
    t = 1 : poisson := solution (target field)

with ``n_steps_input=1, n_steps_output=1``.

Usage
-----
    python data/convert_poisson_to_well.py \\
        --src data/Poisson-Gauss/Poisson-Gauss.nc \\
        --out-dir data_root \\
        --n-samples 10000

Writes under <out-dir>/poisson_gauss/:
    stats.yaml
    data/train/poisson_gauss.hdf5   (8000 samples)
    data/valid/poisson_gauss.hdf5   (1000 samples)
    data/test/poisson_gauss.hdf5    (1000 samples)
"""

import argparse
import os

import h5py
import numpy as np

DATASET_NAME = "poisson_gauss"
FIELD_NAME   = "poisson"
BOUNDARY_TYPE = "wall"   # Dirichlet zero BC


def _write_split_hdf5(out_file: str, data: np.ndarray) -> None:
    n_traj, n_time, nx, ny = data.shape
    assert n_time == 2
    x_coord = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y_coord = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    t_coord = np.linspace(0.0, 1.0, n_time, dtype=np.float32)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with h5py.File(out_file, "w") as dst:
        dst.attrs["grid_type"]     = "cartesian"
        dst.attrs["n_spatial_dims"] = 2
        dst.attrs["dataset_name"]  = DATASET_NAME
        dst.attrs["n_trajectories"] = n_traj

        dims = dst.create_group("dimensions")
        dims.attrs["spatial_dims"] = ["x", "y"]
        for name, arr in [("x", x_coord), ("y", y_coord), ("time", t_coord)]:
            ds = dims.create_dataset(name, data=arr)
            ds.attrs["sample_varying"] = False
            ds.attrs["time_varying"]   = False

        t0 = dst.create_group("t0_fields")
        t0.attrs["field_names"] = [FIELD_NAME]
        field_ds = t0.create_dataset(FIELD_NAME, data=data.astype(np.float32))
        field_ds.attrs["dim_varying"]    = np.array([True, True])
        field_ds.attrs["sample_varying"] = True
        field_ds.attrs["time_varying"]   = True
        field_ds.attrs["symmetric"]      = False

        dst.create_group("t1_fields").attrs["field_names"] = np.array([], dtype="S")
        dst.create_group("t2_fields").attrs["field_names"] = np.array([], dtype="S")
        dst.create_group("scalars").attrs["field_names"]   = np.array([], dtype="S")

        bc = dst.create_group("boundary_conditions")
        for dim in ("x", "y"):
            g = bc.create_group(f"{dim}_boundary")
            g.attrs["bc_type"]         = BOUNDARY_TYPE
            g.attrs["associated_dims"] = [dim]
            g.create_dataset("mask", data=np.ones((nx, ny), dtype=bool))

    print(f"  wrote {out_file}: {n_traj} samples x {n_time} steps x {nx}x{ny}")


def _write_stats_yaml(out_path: str, train: np.ndarray) -> None:
    flat = train.reshape(-1)
    mean = float(flat.mean())
    std  = float(flat.std())
    rms  = float(np.sqrt((flat**2).mean()))

    delta = (train[:, 1:] - train[:, :-1]).reshape(-1)
    mean_delta = float(delta.mean())
    std_delta  = float(delta.std())
    rms_delta  = float(np.sqrt((delta**2).mean()))

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
    print(f"  wrote {out_path}: mean={mean:.4e} std={std:.4e} "
          f"mean_delta={mean_delta:.4e} std_delta={std_delta:.4e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/Poisson-Gauss/Poisson-Gauss.nc")
    parser.add_argument("--out-dir", default="data_root")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Use only the first N samples (default 10000).")
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()

    print(f"Loading {args.src} (first {args.n_samples} samples)")
    with h5py.File(args.src, "r") as f:
        source   = f["source"][:args.n_samples]    # (N, 128, 128)
        solution = f["solution"][:args.n_samples]  # (N, 128, 128)
    print(f"  source {source.shape}, solution {solution.shape}")

    data = np.stack([source, solution], axis=1).astype(np.float32)

    n_total = data.shape[0]
    f_tr, f_va, f_te = args.split
    n_train = int(round(n_total * f_tr))
    n_val   = int(round(n_total * f_va))
    n_test  = n_total - n_train - n_val

    train = data[:n_train]
    valid = data[n_train: n_train + n_val]
    test  = data[n_train + n_val:]
    print(f"  split -> train {n_train}, valid {n_val}, test {n_test}")

    ds_root = os.path.join(args.out_dir, DATASET_NAME)
    _write_split_hdf5(os.path.join(ds_root, "data", "train", f"{DATASET_NAME}.hdf5"), train)
    _write_split_hdf5(os.path.join(ds_root, "data", "valid", f"{DATASET_NAME}.hdf5"), valid)
    _write_split_hdf5(os.path.join(ds_root, "data", "test",  f"{DATASET_NAME}.hdf5"), test)
    _write_stats_yaml(os.path.join(ds_root, "stats.yaml"), train)
    print(f"Done. Dataset root: {ds_root}")


if __name__ == "__main__":
    main()
