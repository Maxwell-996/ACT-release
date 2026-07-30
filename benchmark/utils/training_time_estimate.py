"""Helpers to format durations and estimate total / remaining training wall time."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. ``"2h 15m"``, ``"45s"``)."""
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s" if seconds >= 10 else f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s" if s else f"{m}m"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h {m:02d}m" if m else f"{h}h"
    d, h = divmod(h, 24)
    return f"{d}d {h}h"


def count_epochs_inclusive(start_epoch: int, max_epoch: int) -> int:
    return max(0, max_epoch - start_epoch + 1)


def count_scheduled_phases(
    start_epoch: int,
    max_epoch: int,
    val_frequency: int,
    rollout_val_frequency: int,
) -> tuple[int, int]:
    """How many times ``train()`` runs standard vs rollout validation (including final epoch)."""
    n_val = 0
    n_rollout_val = 0
    for e in range(start_epoch, max_epoch + 1):
        if e % val_frequency == 0 or e == max_epoch:
            n_val += 1
        if e % rollout_val_frequency == 0 or e == max_epoch:
            n_rollout_val += 1
    return n_val, n_rollout_val


def validation_batches_per_pass(
    dataloader_len: int, short_validation_length: int, full: bool
) -> int:
    """Batches processed in one :meth:`Trainer.validation_loop` call."""
    if full:
        return max(dataloader_len, 0)
    return max(0, min(short_validation_length, dataloader_len))


def estimate_forward_steps_training_run(
    *,
    start_epoch: int,
    max_epoch: int,
    train_batches_per_epoch: int,
    val_frequency: int,
    rollout_val_frequency: int,
    val_len: int,
    rollout_val_len: int,
    test_len: int,
    rollout_test_len: int,
    short_validation_length: int,
) -> dict[str, int]:
    """Rough count of train / val rollout-style forward passes (batch = one ``rollout_model`` call).

    Standard and rollout validation use the same inner loop pattern; the last epoch uses
    ``full=True`` (all batches) for val and rollout val. Test passes at the end are always full.
    """
    n_epochs = count_epochs_inclusive(start_epoch, max_epoch)
    n_train = n_epochs * max(train_batches_per_epoch, 0)

    n_val_calls, n_roll_calls = count_scheduled_phases(
        start_epoch, max_epoch, val_frequency, rollout_val_frequency
    )

    n_val_steps = 0
    n_roll_steps = 0
    for e in range(start_epoch, max_epoch + 1):
        full = e == max_epoch
        if e % val_frequency == 0 or e == max_epoch:
            n_val_steps += validation_batches_per_pass(
                val_len, short_validation_length, full
            )
        if e % rollout_val_frequency == 0 or e == max_epoch:
            n_roll_steps += validation_batches_per_pass(
                rollout_val_len, short_validation_length, full
            )

    n_test = validation_batches_per_pass(test_len, short_validation_length, True)
    n_rollout_test = validation_batches_per_pass(
        rollout_test_len, short_validation_length, True
    )

    return {
        "train_rollout_batches": n_train,
        "valid_rollout_batches": n_val_steps,
        "rollout_valid_batches": n_roll_steps,
        "test_rollout_batches": n_test,
        "rollout_test_batches": n_rollout_test,
        "total_rollout_batches": n_train
        + n_val_steps
        + n_roll_steps
        + n_test
        + n_rollout_test,
    }


def linear_eta_seconds(
    *,
    elapsed_seconds: float,
    epochs_completed: int,
    epochs_remaining_after_current: int,
) -> float | None:
    """ETA from a constant mean wall time per completed epoch (``elapsed / completed``)."""
    if epochs_completed <= 0 or epochs_remaining_after_current <= 0:
        return None
    mean = elapsed_seconds / float(epochs_completed)
    return mean * float(epochs_remaining_after_current)


def training_run_step_summary_message(
    *,
    start_epoch: int,
    max_epoch: int,
    train_batches_per_epoch: int,
    val_frequency: int,
    rollout_val_frequency: int,
    val_len: int,
    rollout_val_len: int,
    test_len: int,
    rollout_test_len: int,
    short_validation_length: int,
) -> str:
    """One-line summary for logging before the epoch loop."""
    n_ep = count_epochs_inclusive(start_epoch, max_epoch)
    n_val_calls, n_roll_calls = count_scheduled_phases(
        start_epoch, max_epoch, val_frequency, rollout_val_frequency
    )
    c = estimate_forward_steps_training_run(
        start_epoch=start_epoch,
        max_epoch=max_epoch,
        train_batches_per_epoch=train_batches_per_epoch,
        val_frequency=val_frequency,
        rollout_val_frequency=rollout_val_frequency,
        val_len=val_len,
        rollout_val_len=rollout_val_len,
        test_len=test_len,
        rollout_test_len=rollout_test_len,
        short_validation_length=short_validation_length,
    )
    return (
        f"Run outline: epochs={n_ep}, train_batches/epoch={train_batches_per_epoch}, "
        f"validation_calls={n_val_calls}, rollout_valid_calls={n_roll_calls}, "
        f"~rollout_model batches total={c['total_rollout_batches']} "
        f"(train={c['train_rollout_batches']}, valid={c['valid_rollout_batches']}, "
        f"rollout_valid={c['rollout_valid_batches']}, test={c['test_rollout_batches']}, "
        f"rollout_test={c['rollout_test_batches']})"
    )
