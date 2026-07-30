import logging
import os
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from benchmark.metrics import (
    long_time_metrics,
    make_video,
    plot_all_time_metrics,
    plot_validation_rollout_panel,
    validation_metric_suite,
)
from data.data_formatter import (
    DefaultChannelsFirstFormatter,
    DefaultChannelsLastFormatter,
)
from data.datamodule import AbstractDataModule
from data.datasets import DeltaWellDataset
from data.utils import flatten_field_names
from benchmark.utils.training_time_estimate import (
    format_duration,
    linear_eta_seconds,
    training_run_step_summary_message,
)

logger = logging.getLogger(__name__)


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


def _set_dist_sampler_epoch(dataloader: DataLoader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


class Trainer:
    def __init__(
        self,
        checkpoint_folder: str,
        artifact_folder: str,
        viz_folder: str,
        formatter: str,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        # validation_suite: list,
        epochs: int,
        checkpoint_frequency: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: int,
        make_rollout_videos: bool,
        num_time_intervals: int,
        *,
        enable_rollout_val: bool = False,
        temporal_loss_intervals: list[int] | tuple[int, ...] | None = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        is_distributed: bool = False,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
        max_grad_norm: float | None = 1.0,
        checkpoint_path: str = "",
        **_unused_trainer_kwargs: Any,
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Args:
            checkpoint_folder:
                Path to folder used for storing checkpoints.
            artifact_folder:
                Path to folder used for storing artifacts.
            viz_folder:
                Path to folder used for storing visualizations.
            model:
                The Pytorch model to train
            datamodule:
                A datamodule that provides dataloaders for each split (train, valid, and test)
            optimizer:
                A Pytorch optimizer to perform the backprop (e.g. Adam)
            loss_fn:
                A loss function that evaluates the model predictions to be used for training
            epochs:
                Number of epochs to train the model.
                One epoch correspond to a full loop over the datamodule's training dataloader
            checkpoint_frequency:
                The frequency in terms of number of epochs to save the model checkpoint
            val_frequency:
                The frequency in terms of number of epochs to perform the validation
            rollout_val_frequency:
                The frequency in terms of number of epochs to perform the rollout validation
            max_rollout_steps:
                The maximum number of timesteps to rollout the model
            short_validation_length:
                The number of batches to use for quick intermediate validation during training
            make_rollout_videos:
                A boolean flag to trigger the creation of videos during long rollout validation
            num_time_intervals:
                The number of time intervals to split the loss over (used only when
                ``temporal_loss_intervals`` is ``None``).
            temporal_loss_intervals:
                If set, flat sequence ``[s0, e0, s1, e1, ...]`` defining half-open rollout
                slices ``[s_k, e_k)`` for logged metrics (Python slicing). When ``None``,
                intervals are chosen with log-spaced boundaries from ``num_time_intervals``.
            lr_scheduler:
                A Pytorch learning rate scheduler to update the learning rate during training
            device:
                A Pytorch device (e.g. "cuda" or "cpu")
            is_distributed:
                A boolean flag to trigger DDP training
            enable_amp:
                A boolean flag to enable automatic mixed precision training
            amp_type:
                The type of automatic mixed precision to use. Can be "float16" or "bfloat16"
            max_grad_norm:
                Global L2 gradient norm clip (after ``GradScaler.unscale_`` when AMP fp16 is on).
                Set to ``None`` or ``0`` to disable clipping.
            checkpoint_path:
                The path to the model checkpoint to load. If empty, the model is trained from scratch.
            _unused_trainer_kwargs:
                Absorbs deprecated keys from saved Hydra configs (e.g. ``lambda_tv``).
        """
        self.starting_epoch = 1  # Gets overridden on resume
        self.checkpoint_folder = checkpoint_folder
        self.artifact_folder = artifact_folder
        self.viz_folder = viz_folder
        self.device = device
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.is_delta = isinstance(datamodule.train_dataset_core, DeltaWellDataset)
        self.validation_suite = validation_metric_suite + [self.loss_fn]
        self.max_epoch = epochs
        self.checkpoint_frequency = checkpoint_frequency
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.make_rollout_videos = make_rollout_videos
        self.num_time_intervals = num_time_intervals
        self.enable_rollout_val = enable_rollout_val
        if temporal_loss_intervals is not None:
            tli = tuple(int(x) for x in temporal_loss_intervals)
            if len(tli) % 2 != 0:
                raise ValueError(
                    "temporal_loss_intervals must have even length [s0, e0, s1, e1, ...]; "
                    f"got {len(tli)} values"
                )
            self.temporal_loss_intervals: tuple[int, ...] | None = tli
        else:
            self.temporal_loss_intervals = None
        self.enable_amp = enable_amp
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.grad_scaler = torch.GradScaler(
            self.device.type, enabled=enable_amp and amp_type != "bfloat16"
        )
        self.max_grad_norm = max_grad_norm
        self.is_distributed = is_distributed
        self.best_val_loss = None
        self.starting_val_loss = float("inf")
        self.dset_metadata = self.datamodule.train_dataset_core.metadata
        self.dset_norm = None
        if self.datamodule.train_dataset_core.use_normalization:
            self.dset_norm = self.datamodule.train_dataset_core.norm
        if formatter == "channels_first_default":
            self.formatter = DefaultChannelsFirstFormatter(self.dset_metadata)
        elif formatter == "channels_last_default":
            self.formatter = DefaultChannelsLastFormatter(self.dset_metadata)
        if len(checkpoint_path) > 0:
            self.load_checkpoint(checkpoint_path)

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _is_primary_process(self) -> bool:
        if not self.is_distributed:
            return True
        return dist.is_initialized() and dist.get_rank() == 0

    def _wandb_log(self, data: dict, step: int | None = None) -> None:
        if not self._is_primary_process():
            return
        wandb.log(data, step=step)

    def save_model(self, epoch: int, validation_loss: float, output_path: str):
        """Save the model checkpoint."""
        if self.is_distributed and dist.is_initialized() and dist.get_rank() != 0:
            return
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._unwrap_model().state_dict(),
                "optimizer_state_dit": self.optimizer.state_dict(),
                "validation_loss": validation_loss,
                "best_validation_loss": self.best_val_loss,
            },
            output_path,
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load the model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if self.model is not None:
            self._unwrap_model().load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dit"])
        self.best_val_loss = checkpoint["best_validation_loss"]
        self.starting_val_loss = checkpoint["validation_loss"]
        self.starting_epoch = (
            checkpoint["epoch"] + 1
        )  # Saves after training loop, so start at next epoch

    def normalize(self, batch_dict=None, direct_tensor=None):
        if hasattr(self, "dset_norm") and self.dset_norm:
            if batch_dict is not None:
                batch_dict["input_fields"] = self.dset_norm.normalize_flattened(
                    batch_dict["input_fields"], "variable"
                )
                if "constant_fields" in batch_dict:
                    batch_dict["constant_fields"] = self.dset_norm.normalize_flattened(
                        batch_dict["constant_fields"], "constant"
                    )
            if direct_tensor is not None:
                if self.is_delta:
                    direct_tensor = self.dset_norm.normalize_delta_flattened(
                        direct_tensor, "variable"
                    )
                else:
                    direct_tensor = self.dset_norm.normalize_flattened(
                        direct_tensor, "variable"
                    )
        return batch_dict, direct_tensor

    def denormalize(self, batch_dict=None, direct_tensor=None):
        if hasattr(self, "dset_norm") and self.dset_norm:
            if batch_dict is not None:
                batch_dict["input_fields"] = self.dset_norm.denormalize_flattened(
                    batch_dict["input_fields"], "variable"
                )
                if "constant_fields" in batch_dict:
                    batch_dict["constant_fields"] = (
                        self.dset_norm.denormalize_flattened(
                            batch_dict["constant_fields"], "constant"
                        )
                    )
            if direct_tensor is not None:
                # Delta denormalization is different than full denormalization
                if self.is_delta:
                    direct_tensor = self.dset_norm.delta_denormalize_flattened(
                        direct_tensor, "variable"
                    )
                else:
                    direct_tensor = self.dset_norm.denormalize_flattened(
                        direct_tensor, "variable"
                    )

        return batch_dict, direct_tensor

    def rollout_model(self, model, batch, formatter, train=True):
        """Rollout the model for as many steps as we have data for."""
        inputs, y_ref = formatter.process_input(batch)
        rollout_steps = min(
            y_ref.shape[1], self.max_rollout_steps
        )  # Number of timesteps in target
        y_ref = y_ref[:, :rollout_steps]
        # NOTE: This is a quick fix so we can make datamodule behavior consistent. Revisit this next release (MM).
        # Loss space: ``train=True`` keeps predictions and targets in **dataset-normalized**
        # units for the entire rollout (no denormalize path below). ``train=False`` denormalizes
        # ``y_ref`` once and each ``y_pred`` after every step, so validation/test metrics (e.g.
        # ``H1RelLoss``) are in **physical** units. Raw train vs val loss magnitudes are not
        # directly comparable; train batch logs are also noisier than epoch-averaged curves.
        if not train:
            _, y_ref = self.denormalize(None, y_ref)

        # Create a moving batch of one step at a time
        moving_batch = dict(batch)
        moving_batch["input_fields"] = moving_batch["input_fields"].to(self.device)
        if "constant_fields" in moving_batch:
            moving_batch["constant_fields"] = moving_batch["constant_fields"].to(
                self.device
            )
        y_preds = []
        for i in range(rollout_steps):
            # NOTE: This is a quick fix so we can make datamodule behavior consistent.
            # Including local normalization schemes means there needs to be the option of normalizing each step
            # and there's currently not a registry of local vs global normalization schemes.
            if not train and self.datamodule.val_dataset.use_normalization and i > 0:
                moving_batch, _ = self.normalize(moving_batch)

            inputs, _ = formatter.process_input(moving_batch)
            inputs = [x.to(self.device) for x in inputs]
            raw_out = model(*inputs)
            if isinstance(raw_out, tuple):
                y_pred = raw_out[0]
            else:
                y_pred = raw_out
            y_pred = formatter.process_output_channel_last(y_pred)
            if not train:
                moving_batch, y_pred = self.denormalize(moving_batch, y_pred)

            if (not train) and self.is_delta:
                assert (
                    moving_batch["input_fields"][:, -1, ...].shape == y_pred.shape
                ), f"Mismatching shapes between last input timestep {moving_batch[:, -1, ...].shape}\
                and prediction {y_pred.shape}"
                y_pred = moving_batch["input_fields"][:, -1, ...] + y_pred
            y_pred = formatter.process_output_expand_time(y_pred)
            # If not last step, update moving batch for autoregressive prediction
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred], dim=1
                )
            y_preds.append(y_pred)
        y_pred_out = torch.cat(y_preds, dim=1)
        y_ref = y_ref.to(self.device)
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean()
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) == 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[start_ind:end_ind].mean()
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def _temporal_loss_pairs_clipped(self, time_steps: int) -> list[tuple[int, int]]:
        """Build ``(start, end)`` pairs from ``temporal_loss_intervals``, clipped to ``[0, time_steps]``."""
        raw = self.temporal_loss_intervals
        assert raw is not None
        pairs: list[tuple[int, int]] = []
        for i in range(0, len(raw), 2):
            s, e = int(raw[i]), int(raw[i + 1])
            s = max(0, min(s, time_steps))
            e = max(s, min(e, time_steps))
            if s < e:
                pairs.append((s, e))
        return pairs

    def temporal_split_losses_explicit_pairs(
        self,
        loss_1d: torch.Tensor,
        pairs: list[tuple[int, int]],
        loss_name: str,
        dset_name: str,
        fname: str,
    ) -> dict[str, torch.Tensor]:
        """Per-interval metrics for explicit half-open segments (plus ``T=all``)."""
        new_losses: dict[str, torch.Tensor] = {}
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_1d.mean()
        for s, e in pairs:
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={s}:{e}"] = loss_1d[s:e].mean()
        return new_losses

    def split_up_losses(self, loss_values, loss_name, dset_name, field_names):
        new_losses = {}
        time_logs = {}
        time_steps = loss_values.shape[0]  # we already average over batch

        if self.temporal_loss_intervals is not None:
            pairs = self._temporal_loss_pairs_clipped(time_steps)
            for i, fname in enumerate(field_names):
                time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                    :, i
                ].cpu()
                new_losses |= self.temporal_split_losses_explicit_pairs(
                    loss_values[:, i], pairs, loss_name, dset_name, fname
                )
            new_losses |= self.temporal_split_losses_explicit_pairs(
                loss_values.mean(1), pairs, loss_name, dset_name, "full"
            )
            time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(1).cpu()
            return new_losses, time_logs

        num_time_intervals = min(time_steps, self.num_time_intervals)
        temporal_loss_intervals = np.linspace(0, np.log(time_steps), num_time_intervals)
        temporal_loss_intervals = [0] + [
            int(np.exp(x)) for x in temporal_loss_intervals
        ]
        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, i], temporal_loss_intervals, loss_name, dset_name, fname
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(1), temporal_loss_intervals, loss_name, dset_name, "full"
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(1).cpu()
        return new_losses, time_logs

    @torch.inference_mode()
    def validation_loop(
        self,
        dataloader: DataLoader,
        valid_or_test: str = "valid",
        full: bool = False,
        epoch: int = 0,
    ) -> float:
        """Run validation by looping over the dataloader."""
        self.model.eval()
        validation_loss = 0.0
        field_names = flatten_field_names(self.dset_metadata, include_constants=False)
        dset_name = self.dset_metadata.dataset_name
        viz_split_scope = (
            f"epoch{epoch}/"
            f"{valid_or_test.replace(os.sep, '_').replace('/', '_')}"
        )
        loss_dict = {}
        time_logs = {}
        count = 0
        denom = (
            len(dataloader)
            if full
            else min(self.short_validation_length, len(dataloader))
        )
        with torch.autocast(
            self.device.type, enabled=self.enable_amp, dtype=self.amp_type
        ):
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # Rollout for length of target
                y_pred, y_ref = self.rollout_model(
                    self.model, batch, self.formatter, train=False
                )
                assert (
                    y_ref.shape == y_pred.shape
                ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                batch_loss_scalars: dict[str, float] = {}
                # Go through losses
                for loss_fn in self.validation_suite:
                    # Mean over batch and time per field
                    loss = loss_fn(y_pred, y_ref, self.dset_metadata)
                    # Some losses return multiple values for efficiency
                    if not isinstance(loss, dict):
                        loss = {loss_fn.__class__.__name__: loss}
                    # Split the losses and update the logging dictionary
                    for k, v in loss.items():
                        batch_loss_scalars[f"{loss_fn.__class__.__name__}/{k}"] = (
                            float(v.detach().float().mean().cpu())
                        )
                        sub_loss = v.mean(0)
                        new_losses, new_time_logs = self.split_up_losses(
                            sub_loss, k, dset_name, field_names
                        )
                        # TODO get better way to include spectral error.
                        if k in long_time_metrics or "spectral_error" in k:
                            time_logs |= new_time_logs
                        for loss_name, loss_value in new_losses.items():
                            loss_dict[loss_name] = (
                                loss_dict.get(loss_name, 0.0) + loss_value / denom
                            )
                if self._is_primary_process():
                    try:
                        plot_validation_rollout_panel(
                            y_pred,
                            y_ref,
                            self.dset_metadata,
                            self.viz_folder,
                            epoch,
                            i,
                            sample_index=0,
                            loss_scalars=batch_loss_scalars,
                            split_scope=viz_split_scope,
                        )
                    except Exception:
                        logger.exception(
                            "plot_validation_rollout_panel failed (epoch=%s, "
                            "batch_idx=%s, viz_folder=%s)",
                            epoch,
                            i,
                            self.viz_folder,
                        )
                    if self.make_rollout_videos and y_ref.shape[1] > 1:
                        try:
                            make_video(
                                y_pred[0],
                                y_ref[0],
                                self.dset_metadata,
                                self.viz_folder,
                                epoch,
                                val_dataloader_index=i,
                                split_scope=viz_split_scope,
                            )
                        except Exception:
                            logger.exception(
                                "make_video failed (epoch=%s, batch_idx=%s, "
                                "viz_folder=%s); check ffmpeg and tensor shapes",
                                epoch,
                                i,
                                self.viz_folder,
                            )
                count += 1
                if not full and count >= self.short_validation_length:
                    break

        # Rollout loss curves from the metrics loop (not per-batch).
        if (
            self._is_primary_process()
            and count > 0
            and y_ref.shape[1] > 1
            and time_logs
        ):
            try:
                plot_all_time_metrics(
                    time_logs,
                    self.dset_metadata,
                    self.viz_folder,
                    epoch,
                    split_scope=viz_split_scope,
                )
            except Exception:
                logger.exception(
                    "plot_all_time_metrics failed (epoch=%s, viz_folder=%s)",
                    epoch,
                    self.viz_folder,
                )
            logger.info(
                "Rollout time metrics (numpy): %s/%s/rollout_losses/",
                self.viz_folder,
                viz_split_scope,
            )

        if self._is_primary_process() and count > 0:
            logger.info(
                "Validation rollout panels (per val batch in loop, sample 0, all T): "
                "%s/%s/ (dl*_*.png)",
                self.viz_folder,
                viz_split_scope,
            )

        if self.is_distributed:
            for k in list(loss_dict.keys()):
                v = loss_dict[k]
                if not isinstance(v, torch.Tensor):
                    continue
                t = v.detach().float().to(self.device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                loss_dict[k] = t

        validation_loss = loss_dict[
            f"{dset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
        ].item()
        loss_dict = {f"{valid_or_test}_{k}": v.item() for k, v in loss_dict.items()}
        # Misc metrics
        loss_dict["param_norm"] = param_norm(self.model.parameters())
        return validation_loss, loss_dict

    def train_one_epoch(self, epoch: int, dataloader: DataLoader) -> float:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        train_logs = {}
        start_time = time.time()  # Don't need to sync this.
        batch_start = time.time()
        for i, batch in enumerate(dataloader):
            with torch.autocast(
                self.device.type, enabled=self.enable_amp, dtype=self.amp_type
            ):
                batch_time = time.time() - batch_start
                y_pred, y_ref = self.rollout_model(
                    self.model, batch, self.formatter, train=True
                )
                forward_time = time.time() - batch_start - batch_time
                assert (
                    y_ref.shape == y_pred.shape
                ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                loss = self.loss_fn(y_pred, y_ref, self.dset_metadata).mean()
            self.grad_scaler.scale(loss).backward()
            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.optimizer.zero_grad()
            # Syncing for all reduce anyway so may as well compute synchornous metrics
            epoch_loss += loss.item() / len(dataloader)
            backward_time = time.time() - batch_start - forward_time - batch_time
            total_time = time.time() - batch_start
            logger.info(
                f"Epoch {epoch}, Batch {i + 1}/{len(dataloader)}: loss {loss.item()}, total_time {total_time}, batch time {batch_time}, forward time {forward_time}, backward time {backward_time}"
            )
            batch_start = time.time()
        train_logs["time_per_train_iter"] = (time.time() - start_time) / len(dataloader)
        train_logs["train_loss"] = epoch_loss
        if self.lr_scheduler:
            self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        train_dataloader = self.datamodule.train_dataloader()
        val_dataloder = self.datamodule.val_dataloader()
        rollout_val_dataloader = self.datamodule.rollout_val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()
        rollout_test_dataloader = self.datamodule.rollout_test_dataloader()
        val_loss = self.starting_val_loss

        if self._is_primary_process():
            logger.info(
                training_run_step_summary_message(
                    start_epoch=self.starting_epoch,
                    max_epoch=self.max_epoch,
                    train_batches_per_epoch=len(train_dataloader),
                    val_frequency=self.val_frequency,
                    rollout_val_frequency=self.rollout_val_frequency,
                    val_len=len(val_dataloder),
                    rollout_val_len=len(rollout_val_dataloader),
                    test_len=len(test_dataloader),
                    rollout_test_len=len(rollout_test_dataloader),
                    short_validation_length=self.short_validation_length,
                )
            )

        t_run_start = time.time()
        # Bind ``epoch`` even when the loop body never runs (e.g. resuming a checkpoint
        # whose epoch already reaches ``max_epoch``), so the final test pass below is safe.
        epoch = self.starting_epoch - 1
        for epoch in range(self.starting_epoch, self.max_epoch + 1):
            if self.is_distributed:
                _set_dist_sampler_epoch(train_dataloader, epoch)
                _set_dist_sampler_epoch(val_dataloder, epoch)
                _set_dist_sampler_epoch(rollout_val_dataloader, epoch)
                _set_dist_sampler_epoch(test_dataloader, epoch)
                _set_dist_sampler_epoch(rollout_test_dataloader, epoch)
            # Run training and log training results
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: avg training loss {train_loss}"
            )
            train_logs |= {"train": train_loss, "epoch": epoch}
            self._wandb_log(train_logs, step=epoch)
            # Save the most recent iteration
            self.save_model(
                epoch, val_loss, os.path.join(self.checkpoint_folder, "recent.pt")
            )
            # Check for periodic checkpointing
            if (
                self.checkpoint_frequency >= 1
                and epoch % self.checkpoint_frequency == 0
            ):
                self.save_model(
                    epoch,
                    val_loss,
                    os.path.join(self.checkpoint_folder, f"checkpoint_{epoch}.pt"),
                )
            # Check if time to perform standard validation - periodic or final
            if epoch % self.val_frequency == 0 or (epoch == self.max_epoch):
                logger.info(f"Epoch {epoch}/{self.max_epoch}: starting validation")
                val_loss, val_loss_dict = self.validation_loop(
                    val_dataloder, full=epoch == self.max_epoch, epoch=epoch
                )
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: avg validation loss {val_loss}"
                )
                val_loss_dict |= {"valid": val_loss, "epoch": epoch}
                self._wandb_log(val_loss_dict, step=epoch)
                if self.best_val_loss is None or val_loss < self.best_val_loss:
                    self.save_model(
                        epoch, val_loss, os.path.join(self.checkpoint_folder, "best.pt")
                    )
                    self.best_val_loss = val_loss
            # Check if time for expensive validation - periodic or final. Rollout
            # validation is slow (full-trajectory autoregression); gated off by default
            # so training is fast. Rollout metrics are still reported on the test set below.
            if self.enable_rollout_val and (
                epoch % self.rollout_val_frequency == 0 or (epoch == self.max_epoch)
            ):
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: starting rollout validation"
                )
                rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                    rollout_val_dataloader,
                    valid_or_test="rollout_valid",
                    full=epoch == self.max_epoch,
                    epoch=epoch,
                )
                logger.info(
                    f"Epoch {epoch}/{self.max_epoch}: avg rollout validation loss {rollout_val_loss}"
                )
                rollout_val_loss_dict |= {
                    "rollout_valid": rollout_val_loss,
                    "epoch": epoch,
                }
                self._wandb_log(rollout_val_loss_dict, step=epoch)

            if self._is_primary_process():
                elapsed = time.time() - t_run_start
                epochs_done = epoch - self.starting_epoch + 1
                epochs_remaining_after_current = self.max_epoch - epoch
                eta_sec = linear_eta_seconds(
                    elapsed_seconds=elapsed,
                    epochs_completed=epochs_done,
                    epochs_remaining_after_current=epochs_remaining_after_current,
                )
                if eta_sec is not None:
                    mean_ep = elapsed / float(epochs_done)
                    logger.info(
                        "Training ETA: ~%s wall remaining (mean ~%s/epoch over %d epoch(s); "
                        "elapsed %s)",
                        format_duration(eta_sec),
                        format_duration(mean_ep),
                        epochs_done,
                        format_duration(elapsed),
                    )

        test_loss, test_logs = self.validation_loop(
            test_dataloader,
            valid_or_test="test",
            full=True,
            epoch=epoch + 1000,  # Just use this to flag test
        )
        rollout_test_loss, rollout_test_logs = self.validation_loop(
            rollout_test_dataloader,
            valid_or_test="rollout_test",
            full=True,
            epoch=epoch + 1000,  # Just use this to flag test
        )
        test_logs |= rollout_test_logs
        logger.info(f"Test loss {test_loss}")
        logger.info(f"Rollout test loss {rollout_test_loss}")
        test_logs |= {
            "test": test_loss,
            "rollout_test": rollout_test_loss,
            "epoch": epoch,
        }
        self._wandb_log(test_logs, step=epoch)

    def validate(self):
        """Runs only validation passes - does not save checkpoints or perform training.

        This can probably be refactored to be merged with train, but the flow is already
        convoluted enough that I'm splitting it for now.
        """
        val_dataloder = self.datamodule.val_dataloader()
        rollout_val_dataloader = self.datamodule.rollout_val_dataloader()
        test_dataloader = self.datamodule.test_dataloader()
        rollout_test_dataloader = self.datamodule.rollout_test_dataloader()
        epoch = self.max_epoch + 1
        # Regular val
        val_loss, val_loss_dict = self.validation_loop(
            val_dataloder, full=True, epoch=epoch
        )
        logger.info(f"Post-run: validation loss {val_loss}")
        val_loss_dict |= {"valid": val_loss, "epoch": self.max_epoch + 1}
        self._wandb_log(val_loss_dict, step=epoch)
        # Rollout val
        rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
            rollout_val_dataloader, valid_or_test="rollout_valid", full=True, epoch=epoch
        )
        logger.info(f"Post run: rollout validation loss {rollout_val_loss}")
        rollout_val_loss_dict |= {
            "rollout_valid": rollout_val_loss,
            "epoch": self.max_epoch + 1,
        }
        self._wandb_log(rollout_val_loss_dict, step=epoch)
        # Regular test
        test_loss, test_logs = self.validation_loop(
            test_dataloader, valid_or_test="test", full=False, epoch=epoch
        )
        logger.info(f"Post run: test loss {test_loss}")
        # Rollout test
        rollout_test_loss, rollout_test_logs = self.validation_loop(
            rollout_test_dataloader,
            valid_or_test="rollout_test",
            full=False,
            epoch=epoch,
        )
        test_logs |= rollout_test_logs
        logger.info(f"Post run: rollout test loss {rollout_test_loss}")

        test_logs |= {
            "test": test_loss,
            "rollout_test": rollout_test_loss,
            "epoch": self.max_epoch + 1,
        }
        self._wandb_log(test_logs, step=epoch)
