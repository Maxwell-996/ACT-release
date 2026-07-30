import logging
import os
import os.path as osp
import sys
from datetime import datetime

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hydra
import torch
import torch.distributed as dist
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torchinfo import summary

from benchmark.trainer import Trainer
from benchmark.trainer.utils import get_distrib_config, set_master_config
from benchmark.utils.experiment_utils import configure_experiment
from data import WellDataModule

logger = logging.getLogger("the_well")
logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = osp.join(osp.dirname(__file__), "configs")
CONFIG_NAME = "config"
CONFIG_PATH = osp.join(CONFIG_DIR, f"{CONFIG_NAME}.yaml")
assert osp.isfile(CONFIG_PATH), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")


def train(
    cfg: DictConfig,
    experiment_folder: str,
    checkpoint_folder: str,
    artifact_folder: str,
    viz_folder: str,
    is_distributed: bool = True,
    world_size: int = 1,
    rank: int = 1,
    local_rank: int = 1,
):
    """Instantiate the different objects required for training and run the training loop."""
    validation_mode = cfg.validation_mode
    logger.info(f"Instantiate datamodule {cfg.data._target_}")
    datamodule: WellDataModule = instantiate(
        cfg.data, world_size=world_size, rank=rank, data_workers=cfg.data_workers
    )
    dset_metadata = datamodule.train_dataset_core.metadata
    # TODO - currently just doing channel/time stacking for uniformity, but should
    # give the option of not stacking
    n_input_fields = (
        cfg.data.n_steps_input * dset_metadata.n_fields
        + dset_metadata.n_constant_fields
    )
    n_output_fields = dset_metadata.n_fields

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_spatial_dims=dset_metadata.n_spatial_dims,
        spatial_resolution=dset_metadata.spatial_resolution,
        dim_in=n_input_fields,
        dim_out=n_output_fields,
    )
    summary(model, depth=5)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda")
        model = model.to(device)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=OmegaConf.select(
                cfg, "ddp_find_unused_parameters", default=False
            ),
        )
    else:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model = model.to(device)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if not validation_mode:
        optimizer: torch.optim.Optimizer = instantiate(
            cfg.optimizer, params=model.parameters()
        )
    else:
        optimizer = None

    if hasattr(cfg, "lr_scheduler") and not validation_mode:
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.epochs,
            warmup_start_lr=cfg.optimizer.lr * 0.1,
            eta_min=cfg.optimizer.lr * 0.1,
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None
    # Print final config, but also log it to experiment directory.
    logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        checkpoint_folder=checkpoint_folder,
        artifact_folder=artifact_folder,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        is_distributed=is_distributed,
    )
    if validation_mode:
        trainer.validate()
    else:
        if (not is_distributed) or rank == 0:
            with open(osp.join(experiment_folder, "extended_config.yaml"), "w") as f:
                OmegaConf.save(cfg, f)
        if is_distributed:
            dist.barrier()
        trainer.train()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name=CONFIG_NAME)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.backends.cudnn.benchmark = (
        True  # If input size is fixed, this will usually the computation faster
    )
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    # Normal things
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    ) = configure_experiment(cfg, logger)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    is_distributed, world_size, rank, local_rank = get_distrib_config()
    logger.info(
        f"Distributed training: {is_distributed} "
        f"(world_size={world_size}, rank={rank}, local_rank={local_rank})"
    )

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP is configured but CUDA is not available.")
        set_master_config()
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(local_rank))
        dist.init_process_group(backend="nccl", init_method="env://")

    wandb_logged_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb_logged_cfg["experiment_folder"] = experiment_folder
    wandb_run_name = experiment_name
    if OmegaConf.select(cfg, "wandb_append_launch_timestamp", default=False):
        wandb_run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0:
        logger.info("WandB run name: %s", wandb_run_name)
        wandb.init(
            dir=experiment_folder,
            project=cfg.wandb_project_name,
            group=f"{cfg.data.well_dataset_name}",
            config=wandb_logged_cfg,
            name=wandb_run_name,
            resume=True,
        )
    else:
        wandb.init(mode="disabled")

    try:
        train(
            cfg,
            experiment_folder,
            checkpoint_folder,
            artifact_folder,
            viz_folder,
            is_distributed,
            world_size,
            rank,
            local_rank,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        wandb.finish()


if __name__ == "__main__":
    main()
