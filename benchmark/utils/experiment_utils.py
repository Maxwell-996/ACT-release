import logging
import os
import os.path as osp
import time
from datetime import datetime
from typing import Tuple

from omegaconf import DictConfig, OmegaConf


def _list_run_subdirs(base_experiment_folder: str) -> list[str]:
    """Names of direct child directories under ``base_experiment_folder``."""
    if not osp.isdir(base_experiment_folder):
        return []
    return sorted(
        name
        for name in os.listdir(base_experiment_folder)
        if osp.isdir(osp.join(base_experiment_folder, name))
    )


def _latest_run_dir(base_experiment_folder: str) -> str | None:
    """Full path to the child run directory with the newest modification time, or ``None``."""
    names = _list_run_subdirs(base_experiment_folder)
    if not names:
        return None
    latest_name = max(
        names,
        key=lambda d: osp.getmtime(osp.join(base_experiment_folder, d)),
    )
    return osp.join(base_experiment_folder, latest_name)


def _new_timestamped_run_dir(base_experiment_folder: str) -> str:
    """Return ``.../base/<YYYYMMDD_HHMMSS>`` or ``..._<n>`` if that name already exists."""
    os.makedirs(base_experiment_folder, exist_ok=True)
    base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_name
    n = 0
    while osp.exists(osp.join(base_experiment_folder, candidate)):
        n += 1
        candidate = f"{base_name}_{n}"
    return osp.join(base_experiment_folder, candidate)


def _torchrun_rank_world() -> tuple[int, int]:
    """``(rank, world_size)`` from the launcher env before ``init_process_group``."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])


def _allocate_run_dir_ddp(
    base_experiment_folder: str, logger: logging.Logger
) -> str:
    """Single new run directory for all ranks (torchrun / DDP).

    Rank 0 picks a fresh timestamped path and writes it to a marker file keyed by this
    process group's rendezvous; other ranks poll the same file. Avoids each rank creating
    its own subdirectory under ``base_experiment_folder``.
    """
    rank, world_size = _torchrun_rank_world()
    if world_size <= 1:
        return _new_timestamped_run_dir(base_experiment_folder)

    os.makedirs(base_experiment_folder, exist_ok=True)
    slot = (
        f"{os.environ.get('MASTER_ADDR', 'localhost')}_"
        f"{os.environ.get('MASTER_PORT', '0')}_"
        f"{os.getppid()}"
    )
    stamp = osp.join(base_experiment_folder, f".ddp_experiment_dir_{slot}.txt")
    tmp = stamp + ".tmp"

    if rank == 0:
        chosen = _new_timestamped_run_dir(base_experiment_folder)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(chosen)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, stamp)
        logger.info("DDP: rank 0 chose experiment_folder=%s", chosen)
        return chosen

    deadline = time.time() + 600.0
    while time.time() < deadline:
        if osp.isfile(stamp):
            with open(stamp, encoding="utf-8") as f:
                chosen = f.read().strip()
            if chosen:
                logger.info("DDP: rank %s using experiment_folder=%s", rank, chosen)
                return chosen
        time.sleep(0.05)
    raise RuntimeError(
        f"Timed out waiting for rank 0 to write experiment path marker {stamp}"
    )


def configure_paths(experiment_folder):
    """Configure the paths for the experiment with the given experiment folder."""
    # Make checkpoints directory as experiment_folder/checkpoints
    os.makedirs(osp.join(experiment_folder, "checkpoints"), exist_ok=True)
    checkpoint_folder = osp.join(experiment_folder, "checkpoints")
    # Store run/validation data as experiment_folder/artifacts
    os.makedirs(osp.join(experiment_folder, "artifacts"), exist_ok=True)
    artifact_folder = osp.join(experiment_folder, "artifacts")
    # Plot data directory as experiment_folder/viz
    os.makedirs(osp.join(experiment_folder, "viz"), exist_ok=True)
    viz_folder = osp.join(experiment_folder, "viz")
    return checkpoint_folder, artifact_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    model_name = cfg.model._target_.split(".")[-1]
    # slurm_job_id = os.environ.get("SLURM_JOB_ID", "0") - Not using for now since I think it'll be easier to just use name alone
    return f"{cfg.data.well_dataset_name}-{cfg.name}-{model_name}-{cfg.optimizer.lr}"


def configure_experiment(
    cfg: DictConfig, logger: logging.Logger
) -> Tuple[DictConfig, str, str]:
    """Works through resume logic to figure out where to save the current experiment
    and where to look to resume or validate previous experiments.

    If the user provides overrides for the folder/checkpoint/config, use them.

    If folder isn't provided, construct default. New runs use a timestamp subdirectory
    (with optional numeric suffix if the name exists). Under torchrun/DDP, only rank 0
    picks the name and broadcasts it via a marker file so all processes share one folder.
    If autoresume or validation_mode is enabled, the run with the newest directory mtime
    under that experiment is chosen (works with legacy numeric run ids and timestamps).

    If checkpoint is provided, use it to override any weights obtained until now. If
    any checkpoint is available either in the folder or checkpoint override, this
    is considered a resume run.

    If it's in validation mode but no checkpoint is found, throw an error.

    If config override is provided, use it (with the weights and current output folder).
    Otherwise start search over hierarchy.
      - If checkpoint is being used, look to see if it has an associated config file
      - If no checkpoint but folder, look in folder
      - If not, just use the default config (whatever is currently set)



    Args:
        experiment_folder:
            Path to base folder used for experiment
        logger:
            Logger object to print messages to console
    """
    # Sort out default names and folders
    experiment_name = get_experiment_name(cfg)
    base_experiment_folder = osp.join(cfg.experiment_dir, experiment_name)
    experiment_folder = cfg.folder_override  # Default is ""
    checkpoint_file = cfg.checkpoint_override  # Default is ""
    config_file = cfg.config_override  # Default is ""
    validation_mode = cfg.validation_mode
    # If using default naming: timestamped new dir, or latest subdir for val/auto_resume.
    if len(experiment_folder) == 0:
        latest = _latest_run_dir(base_experiment_folder)
        if (validation_mode or cfg.auto_resume) and latest is not None:
            experiment_folder = latest
        elif validation_mode:
            raise ValueError(
                f"Validation mode enabled but no previous runs found in {base_experiment_folder}."
            )
        else:
            experiment_folder = _allocate_run_dir_ddp(base_experiment_folder, logger)
    # Now check for default checkpoint options - if override used, ignore
    if osp.exists(experiment_folder) and len(checkpoint_file) == 0:
        last_chpt = osp.join(experiment_folder, "checkpoints", "recent.pt")
        # If there's a checkpoint file, consider this a resume. Otherwise, this is new run.
        if osp.isfile(last_chpt):
            checkpoint_file = last_chpt
    if len(checkpoint_file) > 0:
        logger.info(f"Checkpoint found, using checkpoint file {checkpoint_file}")
    if not osp.isfile(checkpoint_file) and len(checkpoint_file) > 0:
        raise ValueError(
            f"Checkpoint path provided but checkpoint file {checkpoint_file} not found."
        )
    # Now pick a config file to use - either current, override, or related to a different override
    if len(checkpoint_file) > 0 and len(config_file) == 0:
        # Check two levels - the parent folder of the checkpoint and the experiment folder
        checkpoint_path = osp.join(
            osp.dirname(checkpoint_file), osp.pardir, "extended_config.yaml"
        )
        folder_path = osp.join(experiment_folder, "extended_config.yaml")
        if osp.isfile(checkpoint_path):
            logger.info(
                f"Config file exists relative to checkpoint override provided, \
                            using config file {checkpoint_path}"
            )
        elif osp.isfile(folder_path):
            logger.warn(
                f"Config file not found in checkpoint override path. \
                        Found in experiment folder, using config file {folder_path}. \
                        This could lead to weight compatibility issues if the checkpoints do not align with \
                        the specified folder."
            )
        else:
            logger.warn(
                "Checkpoint override provided, but config file not found in checkpoint override path \
                        or experiment folder. Using default configuration which may not be compatible with checkpoint."
            )
        # resume = True
    elif len(config_file) > 0:
        logger.info(f"Config override provided, using config file {config_file}")
    elif validation_mode:
        raise ValueError(
            f"Validation mode enabled but no checkpoint provided or found in {experiment_folder}."
        )
    if len(config_file) > 0:
        cfg = OmegaConf.load(config_file)
    cfg.trainer.checkpoint_path = checkpoint_file
    # cfg.trainer.resume = resume
    # Create experiment folder if it doesn't already exist
    os.makedirs(experiment_folder, exist_ok=True)
    checkpoint_folder, artifact_folder, viz_folder = configure_paths(experiment_folder)
    return (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    )
