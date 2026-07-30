#!/usr/bin/env python
"""Convert the raw ``ns`` vorticity tensor into The Well HDF5 format.

Source
------
``ns_128_T50_0.0001/ns_128_T50_0.0001.pt`` is a single ``float32`` tensor of shape
``[500, 100, 128, 128]`` (500 trajectories x 100 timesteps x 128 x 128 vorticity field).

Processing (mirrors the reference ``split_train_val_test`` pipeline)
--------------------------------------------------------------------
1. Temporal downsample ``data[:, 10::2]`` -> 45 frames per trajectory (drop the first
   10 steps, take every 2nd step). Training later slides a window over these frames.
2. Split **by trajectory**: first 400 -> train, next 50 -> valid, last 50 -> test.
3. Write one Well HDF5 file per split with a single scalar (``t0``) field ``vorticity``
   (sample- and time-varying), a cartesian x/y grid, a time axis and periodic boundaries.
4. Emit ``stats.yaml`` from the **train** split (mean/std + delta stats + rms keys) so that
   :class:`data.normalization.ZScoreNormalization` can normalize the data.

The layout produced under ``--out-dir`` is::

    <out-dir>/ns/
        stats.yaml
        data/train/ns.hdf5
        data/valid/ns.hdf5
        data/test/ns.hdf5

which matches ``WellDataset(well_base_path=<out-dir>, well_dataset_name="ns")``.
"""

import argparse
import os

import h5py
import numpy as np
import torch

DATASET_NAME = "ns"
FIELD_NAME = "vorticity"
BOUNDARY_TYPE = "periodic"


def _write_split_hdf5(out_file: str, data: np.ndarray) -> None:
    """Write one split to a Well-format HDF5 file.

    Args:
        out_file: Destination ``.hdf5`` path.
        data: ``(n_traj, n_time, nx, ny)`` float32 array for this split.
    """
    n_traj, n_time, nx, ny = data.shape
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

    print(f"  wrote {out_file}: {n_traj} traj x {n_time} steps x {nx}x{ny}")


def _write_stats_yaml(out_path: str, train: np.ndarray) -> None:
    """Compute normalization statistics from the train split and write ``stats.yaml``.

    ``ZScoreNormalization`` requires the keys ``mean``, ``std``, ``mean_delta``,
    ``std_delta``; we also emit ``rms``/``rms_delta`` for compatibility with RMS
    normalization. Deltas are consecutive-frame differences along time.
    """
    flat = train.reshape(-1)
    mean = float(flat.mean())
    std = float(flat.std())
    rms = float(np.sqrt((flat**2).mean()))

    delta = (train[:, 1:] - train[:, :-1]).reshape(-1)
    mean_delta = float(delta.mean())
    std_delta = float(delta.std())
    rms_delta = float(np.sqrt((delta**2).mean()))

    def _fmt(section: dict) -> str:
        return "\n".join(f"  {FIELD_NAME}: {v:.8E}" for v in section.values())

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
    parser = argparse.ArgumentParser(description="Convert ns .pt tensor to The Well HDF5")
    parser.add_argument(
        "--src",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "ns_128_T50_0.0001",
            "ns_128_T50_0.0001.pt",
        ),
        help="Path to the raw ns .pt tensor [500, 100, 128, 128].",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_root"
        ),
        help="Base output dir; dataset is written under <out-dir>/ns.",
    )
    parser.add_argument("--n-train", type=int, default=400)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument(
        "--time-start", type=int, default=10, help="Drop steps before this index."
    )
    parser.add_argument(
        "--time-stride", type=int, default=2, help="Keep every k-th step after start."
    )
    args = parser.parse_args()

    print(f"Loading {args.src}")
    data = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(data)}")
    data = data.numpy()
    print(f"  raw shape {data.shape}")

    # 1. Temporal downsample: drop the first ``time_start`` steps, take every stride-th.
    data = data[:, args.time_start :: args.time_stride]
    print(f"  after downsample [:, {args.time_start}::{args.time_stride}] -> {data.shape}")

    n_total = data.shape[0]
    n_needed = args.n_train + args.n_val + args.n_test
    if n_needed > n_total:
        raise ValueError(
            f"Requested {n_needed} trajectories but only {n_total} available."
        )

    # 2. Split by trajectory (deterministic prefix ordering).
    train = data[: args.n_train]
    valid = data[args.n_train : args.n_train + args.n_val]
    test = data[args.n_train + args.n_val : args.n_train + args.n_val + args.n_test]
    print(
        f"  split by trajectory -> train {train.shape[0]}, "
        f"valid {valid.shape[0]}, test {test.shape[0]}"
    )

    ds_root = os.path.join(args.out_dir, DATASET_NAME)
    _write_split_hdf5(os.path.join(ds_root, "data", "train", "ns.hdf5"), train)
    _write_split_hdf5(os.path.join(ds_root, "data", "valid", "ns.hdf5"), valid)
    _write_split_hdf5(os.path.join(ds_root, "data", "test", "ns.hdf5"), test)
    _write_stats_yaml(os.path.join(ds_root, "stats.yaml"), train)

    print(f"Done. Dataset root: {ds_root}")


if __name__ == "__main__":
    main()
