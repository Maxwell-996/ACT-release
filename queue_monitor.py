#!/usr/bin/env python3
"""GPU-watching training queue for DarcyFlow ACT/vanilla baselines.

Watches the two GPU pairs (0,1) and (2,3). Whenever a pair is idle it launches
the next pending model on that pair via 2-GPU DDP (torchrun). Runs until every
queued model has trained to completion.

Queue (act_fno already done):
    vanilla_fno_darcy
    vanilla_transolver_darcy
    vanilla_unet_convnext_darcy
    act_transolver_darcy
    act_convnext_darcy

Restart-safe: a model whose ``.done`` marker exists is skipped. Launch it with
``nohup ... &`` and tail ``logs/queue_monitor.log``.
"""

import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.environ.get("ACT_ROOT", os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("THEWELL_PYTHON", "python")
LOG_DIR = os.path.join(ROOT, "logs")
DONE_DIR = os.path.join(ROOT, "experiments", ".queue_done")

# Ordered queue: (experiment_config_name, per-gpu batch size).
QUEUE = [
    ("vanilla_fno_darcy", 64),
    ("vanilla_transolver_darcy", 64),
    ("vanilla_unet_convnext_darcy", 64),
    ("act_transolver_darcy", 64),
    ("act_convnext_darcy", 64),
]

PAIRS = [(0, 1), (2, 3)]
EPOCHS = 300
WORKERS = 8
POLL_SEC = 30          # seconds between GPU polls
FREE_MEM_MIB = 2000    # a GPU counts as free below this much used memory
FREE_UTIL_PCT = 10     # ... and below this utilization
STABLE_POLLS = 2       # pair must look free this many consecutive polls
BASE_PORT = 29520


def log(msg: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def gpu_stats():
    """Return {index: (mem_used_MiB, util_pct)} from nvidia-smi."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    stats = {}
    for line in out.strip().splitlines():
        idx, mem, util = (p.strip() for p in line.split(","))
        stats[int(idx)] = (int(mem), int(util))
    return stats


def pair_is_free(pair, stats) -> bool:
    for g in pair:
        if g not in stats:
            return False
        mem, util = stats[g]
        if mem > FREE_MEM_MIB or util > FREE_UTIL_PCT:
            return False
    return True


def done_marker(name: str) -> str:
    return os.path.join(DONE_DIR, f"{name}.done")


def launch(name: str, batch: int, pair, port: int):
    """Spawn a training run on ``pair`` and return the Popen handle."""
    gpus = ",".join(str(g) for g in pair)
    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["CUDA_VISIBLE_DEVICES"] = gpus
    log_path = os.path.join(LOG_DIR, f"{name}.log")
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        f"--master_port={port}",
        "benchmark/train.py",
        f"experiment={name}",
        "server=local",
        f"trainer.epochs={EPOCHS}",
        f"data.batch_size={batch}",
        f"data_workers={WORKERS}",
    ]
    log(f"LAUNCH {name} on GPUs {gpus} (port {port}, batch {batch}) -> logs/{name}.log")
    logf = open(log_path, "w")
    # Wrap so we can drop a .done marker only on clean exit (rc == 0).
    marker = done_marker(name)
    inner = " ".join(cmd)
    wrapped = f"{inner}; rc=$?; if [ $rc -eq 0 ]; then touch '{marker}'; fi; exit $rc"
    proc = subprocess.Popen(
        wrapped, shell=True, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT
    )
    return proc


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)

    pending = [(n, b) for (n, b) in QUEUE if not os.path.exists(done_marker(n))]
    skipped = [n for (n, _) in QUEUE if os.path.exists(done_marker(n))]
    if skipped:
        log(f"Skipping already-done: {skipped}")
    log(f"Pending queue ({len(pending)}): {[n for n, _ in pending]}")

    # pair -> running (name, Popen) or None
    running = {pair: None for pair in PAIRS}
    stable = {pair: 0 for pair in PAIRS}
    port = BASE_PORT

    while pending or any(running.values()):
        # 1. Reap finished jobs.
        for pair, job in list(running.items()):
            if job is None:
                continue
            name, proc = job
            rc = proc.poll()
            if rc is None:
                continue
            if rc == 0:
                log(f"DONE {name} on GPUs {pair} (rc=0)")
            else:
                log(f"FAILED {name} on GPUs {pair} (rc={rc}) -- see logs/{name}.log; NOT requeued")
            running[pair] = None
            stable[pair] = 0

        if not pending:
            if any(running.values()):
                time.sleep(POLL_SEC)
                continue
            break

        # 2. Poll GPUs and launch on any free, idle pair.
        try:
            stats = gpu_stats()
        except Exception as e:  # nvidia-smi hiccup: skip this poll
            log(f"nvidia-smi error: {e}")
            time.sleep(POLL_SEC)
            continue

        for pair in PAIRS:
            if not pending:
                break
            if running[pair] is not None:
                continue
            if pair_is_free(pair, stats):
                stable[pair] += 1
            else:
                stable[pair] = 0
            if stable[pair] >= STABLE_POLLS:
                name, batch = pending.pop(0)
                port += 1
                proc = launch(name, batch, pair, port)
                running[pair] = (name, proc)
                stable[pair] = 0
                # Give the new job a head start so its GPUs read as busy
                # before the next poll (avoids double-assigning the pair).
                time.sleep(45)

        time.sleep(POLL_SEC)

    log("All queued models finished. Monitor exiting.")


if __name__ == "__main__":
    main()
