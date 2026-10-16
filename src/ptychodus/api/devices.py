"""Enumerate the compute devices a spawned child can use, and restrict it to a few.

Device selection is placement, not configuration. Which GPU a reconstruction
runs on says nothing about the algorithm it runs, and the process that spawns
the child is the one in a position to decide it -- so the decision lives here,
beside the spawn, and a backend neither declares nor receives it.

A device is named by an opaque identifier rather than an integer index, because
the index alone cannot say which runtime is being indexed: builds exist for
CUDA, ROCm and XPU, and torch presents the first two through the same interface.
The identifier carries the family, which is what decides the variable its
runtime reads.

**Restriction is a mask over a mask.** A probe inherits whatever mask the parent
was launched under, so the ordinals it reports are positions *within* that mask:
started under ``CUDA_VISIBLE_DEVICES=1``, it sees one device and calls it zero.
Writing that zero into a child would send it to the device the scheduler did not
grant. Every ordinal is therefore resolved back through the inherited mask
before it is handed on, and entries are copied rather than parsed so that a mask
written as UUIDs composes the same way as one written as indices.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any, Final
import logging
import os
import pickle

from .spawn import ChildError, run_subprocess

__all__ = [
    'Device',
    'probe_devices',
    'visibility_environment',
]

logger = logging.getLogger(__name__)

_PROBE_ENTRY = 'ptychodus.api.devices:_probe_devices_child'
_PROBE_TAG = 'devices'


@dataclass(frozen=True)
class Device:
    """One device a child can be restricted to.

    ``identifier`` is what a selection records and what :func:`visibility_environment`
    reads; ``name`` is for a person choosing between two of them.
    """

    identifier: str
    name: str


# Identifier family -> the variable its runtime reads to decide which devices
# exist. torch exposes ROCm through its CUDA interface, so the two present
# identically and are told apart by the build rather than by the API; the probe
# resolves that once and spells the difference into the identifier.
_VISIBILITY_VARIABLES: Final[Mapping[str, str]] = {
    'cuda': 'CUDA_VISIBLE_DEVICES',
    'rocm': 'HIP_VISIBLE_DEVICES',
    'xpu': 'ZE_AFFINITY_MASK',
}


def _enumerate_devices() -> list[Device]:
    """Return the devices torch can see. Runs in a child: this imports torch."""
    import torch

    devices: list[Device] = []

    if torch.cuda.is_available():
        family = 'rocm' if torch.version.hip else 'cuda'

        for index in range(torch.cuda.device_count()):
            devices.append(Device(f'{family}:{index}', torch.cuda.get_device_name(index)))

    xpu = getattr(torch, 'xpu', None)

    if xpu is not None and xpu.is_available():
        for index in range(xpu.device_count()):
            devices.append(Device(f'xpu:{index}', xpu.get_device_name(index)))

    return devices


def _probe_devices_child(_payload: Any, queue: Queue[Any]) -> None:
    """Child entry point: enumerate devices, put them on the queue, exit."""
    queue.put((_PROBE_TAG, pickle.dumps(_enumerate_devices())))


def probe_devices(*, terminate_grace_sec: float = 5.0) -> Sequence[Device]:
    """Return the devices a child spawned from this process would see.

    Asked in a throwaway child, because asking it here would leave a GPU
    context in a process that must never hold one -- and a child spawned from a
    process holding one is a known source of initialization failures.

    Every failure is non-fatal and yields an empty sequence: a broken GPU stack
    should cost a user the device list, not the application.
    """
    try:
        with run_subprocess(_PROBE_ENTRY, None, terminate_grace_sec=terminate_grace_sec) as events:
            for event in events:
                if event[0] == _PROBE_TAG:
                    devices = pickle.loads(event[1])

                    if isinstance(devices, list):
                        return devices
    except ChildError as exc:
        logger.warning('Device probe subprocess failed: %s', exc.child_exception_type)
    except Exception:
        logger.exception('Device probe subprocess raised in the parent.')

    return []


def visibility_environment(
    identifiers: Sequence[str], *, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the environment entries that restrict a child to *identifiers*.

    Empty when nothing is selected, which leaves the child seeing whatever this
    process does. An identifier naming a family with no mask of its own, or one
    this process cannot reach, contributes nothing rather than raising: a
    selection outlives the hardware it was made against.
    """
    environ = os.environ if environ is None else environ
    selected: dict[str, list[str]] = {}

    for identifier in identifiers:
        family, _, ordinal = identifier.partition(':')

        try:
            variable = _VISIBILITY_VARIABLES[family]
        except KeyError:
            logger.debug(f'Device "{identifier}" names no maskable family; ignored.')
            continue

        entry = _resolve_through_inherited_mask(ordinal, environ.get(variable, ''))

        if entry is None:
            logger.warning(
                f'Device "{identifier}" is not visible to this process; '
                'it is dropped from the selection.'
            )
            continue

        selected.setdefault(variable, []).append(entry)

    return {variable: ','.join(entries) for variable, entries in selected.items()}


def _resolve_through_inherited_mask(ordinal: str, inherited_mask: str) -> str | None:
    """Turn a probed ordinal into a value the child's runtime reads the same way."""
    inherited = [entry.strip() for entry in inherited_mask.split(',') if entry.strip()]

    if not inherited:
        # The two numberings coincide, so the ordinal passes through unparsed
        # and a probe that reported something other than an integer still works.
        return ordinal or None

    try:
        index = int(ordinal)
    except ValueError:
        return None

    return inherited[index] if 0 <= index < len(inherited) else None
