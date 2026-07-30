"""Utility functions to retrieve and set environment variables related to SLURM.
Largely taken from the `idr_torch` module on Jean-Zay.

"""

import logging
import os
from typing import List, Tuple, Union

logger = logging.getLogger(__name__)


def nodelist() -> Union[List[str], str]:
    compact_nodelist = os.environ["SLURM_STEP_NODELIST"]
    try:
        from hostlist import expand_hostlist
    except ImportError:
        return compact_nodelist
    else:
        return expand_hostlist(compact_nodelist)


def get_first_host(hostlist: str) -> str:
    """
    Get the first host from SLURM's nodelist.
    Example: Nodelist="Node[1-5],Node7" -> First node: "Node1"

    Args:
        The compact nodelist as given by SLURM
    Returns:
        The first node to host the master process
    """
    from re import findall, split, sub

    regex = r"\[([^[\]]*)\]"
    all_replacement: list[str] = findall(regex, hostlist)
    new_values = [split("-|,", element)[0] for element in all_replacement]
    for i in range(len(new_values)):
        hostlist = sub(regex, new_values[i], hostlist, count=1)
    return hostlist.split(",")[0]


def get_master_address() -> str:
    nodes = nodelist()
    if isinstance(nodes, list):
        return nodes[0]
    return get_first_host(nodes)


def get_master_port() -> str:
    job_id = int(os.environ["SLURM_JOB_ID"])
    return str(1000 + job_id % 2000)


def get_distrib_config() -> Tuple[bool, int, int, int]:
    """Detect multi-process training (``torchrun`` / torch elastic or SLURM).

    Returns:
        ``(is_distributed, world_size, rank, local_rank)``.
        ``is_distributed`` is True only when ``world_size > 1``.
    """
    # torchrun / torch.distributed.launch set these on every worker
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if world_size <= 1:
            logger.debug("WORLD_SIZE<=1; single-process training.")
            return False, 1, 0, 0
        logger.debug(
            f"Torch distributed env detected, rank {rank} (local {local_rank})/{world_size}"
        )
        return True, world_size, rank, local_rank

    distrib_env_variables = ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_STEP_NUM_TASKS"]
    if not (set(distrib_env_variables) <= set(os.environ)):
        logger.debug("No torch or Slurm distributed configuration in the environment")
        return False, 1, 0, 0

    world_size = int(os.environ["SLURM_STEP_NUM_TASKS"])
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    if world_size <= 1:
        logger.debug("Slurm single task; single-process training.")
        return False, 1, 0, 0
    logger.debug(
        f"Slurm configuration detected, rank {rank} (local {local_rank})/{world_size}"
    )
    return True, world_size, rank, local_rank


def set_master_config() -> None:
    """Set ``MASTER_ADDR`` / ``MASTER_PORT`` when missing (SLURM or local multi-GPU)."""
    if os.environ.get("MASTER_ADDR") and os.environ.get("MASTER_PORT"):
        logger.debug("MASTER_ADDR / MASTER_PORT already set; leaving unchanged.")
        return
    if "SLURM_STEP_NODELIST" in os.environ:
        master_address = get_master_address()
        master_port = get_master_port()
        logger.debug(f"Set master address to {master_address} and port to {master_port}")
        os.environ["MASTER_ADDR"] = master_address
        os.environ["MASTER_PORT"] = master_port
        return
    # e.g. multi-GPU on one machine without torchrun pre-export
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    logger.debug(
        f"Default master {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )
