"""What a distributed launcher says about this process, read from the environment alone.

``torchrun`` and ``torch.distributed.launch`` export the torch names; a plain ``srun``
exports only the SLURM ones. Each reader consults the torch name first, because a job
launched through ``srun`` under ``torchrun`` sets both and the torch one is what tracks the
process group a backend will build.

Nothing here imports torch or any other optional extra. The environment is the only source
a process has before it builds a process group, and the only one at all for a process that
leaves the group to a child it spawns.
"""

from __future__ import annotations

from collections.abc import Sequence
import os

__all__ = [
    'environment_node_count',
    'environment_rank',
    'environment_world_size',
]


def _first_integer(names: Sequence[str], default: int) -> int:
    """The first of `names` whose value parses as an integer, or `default` when none does.

    A value that is empty or not an integer is skipped rather than fatal: it means
    something in the environment claims a value it cannot state, and the next name, or the
    single-process default, is the better reading.
    """
    for name in names:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            continue

    return default


def environment_rank() -> int:
    """This process's global rank as the launcher reported it, or 0 when no launcher did."""
    return _first_integer(('RANK', 'SLURM_PROCID'), 0)


def environment_world_size() -> int:
    """How many processes the launcher started in all, or 1 when no launcher did."""
    return _first_integer(('WORLD_SIZE', 'SLURM_NTASKS'), 1)


def environment_node_count() -> int:
    """How many nodes the launcher spans, or 1 when no launcher did.

    ``torchrun`` reports it as ``GROUP_WORLD_SIZE``, the number of its per-node agents.
    """
    return _first_integer(('GROUP_WORLD_SIZE', 'SLURM_NNODES'), 1)
