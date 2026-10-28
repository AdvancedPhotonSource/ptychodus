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

**The CPU is a device in this list.** Choosing where to run and choosing whether
to use an accelerator at all are the same question asked once, so a selection
naming only the CPU masks every accelerator away and a backend that asks its
framework what it can see finds nothing. That is what lets a backend hold no
device option of its own.

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
from multiprocessing.queues import SimpleQueue
from typing import Any, Final
import logging
import os

from .spawn import ChildError, run_subprocess

__all__ = [
    'CPU',
    'Device',
    'enumerate_devices',
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


# Always present, and the only way a selection says 'no accelerator'.
CPU = Device('cpu', 'CPU')


def enumerate_devices() -> list[Device]:
    """Return the devices torch can see, **by asking it in this process**.

    Importing torch and querying it may leave this process holding a GPU
    context, which is fatal to a long-lived parent: it pins driver state and
    memory, and a child spawned afterwards can fail to initialize. Call this
    only from a process that is about to exit anyway, such as a one-shot
    diagnostic or a child spawned for the purpose. A process that outlives the
    call wants :func:`probe_devices` instead.

    Shared so that what counts as a device, and how it is named, is decided in
    one place: the family spelled into an identifier is resolved here, where
    the torch build can be inspected.

    torch is an optional dependency, and its absence is an answer rather than a
    failure: a machine without it has exactly one device, and saying so keeps a
    bare install able to express where a run should go.
    """
    devices: list[Device] = [CPU]

    try:
        import torch
    except ImportError:
        return devices

    if torch.cuda.is_available():
        family = 'rocm' if torch.version.hip else 'cuda'

        for index in range(torch.cuda.device_count()):
            devices.append(Device(f'{family}:{index}', torch.cuda.get_device_name(index)))

    xpu = getattr(torch, 'xpu', None)

    if xpu is not None and xpu.is_available():
        for index in range(xpu.device_count()):
            devices.append(Device(f'xpu:{index}', xpu.get_device_name(index)))

    return devices


def _probe_devices_child(_payload: Any, queue: SimpleQueue[Any]) -> None:
    """Child entry point: enumerate devices, put them on the queue, exit."""
    queue.put((_PROBE_TAG, enumerate_devices()))


def probe_devices(*, terminate_grace_sec: float = 5.0) -> Sequence[Device]:
    """Return the devices a child spawned from this process would see.

    Asked in a throwaway child running :func:`enumerate_devices`, because
    asking it here would leave a GPU context in a process that must never hold
    one -- and a child spawned from a process holding one is a known source of
    initialization failures.

    Always reports at least the CPU, so a selection can still say where a run
    should go. Every failure is non-fatal: a broken GPU stack should cost a
    user the accelerators it could not enumerate, not the application, and not
    the one device that is there whatever happens.
    """
    try:
        with run_subprocess(_PROBE_ENTRY, None, terminate_grace_sec=terminate_grace_sec) as events:
            for event in events:
                if event[0] == _PROBE_TAG:
                    devices = event[1]

                    if isinstance(devices, list):
                        return devices
    except ChildError as exc:
        logger.warning('Device probe subprocess failed: %s', exc.child_exception_type)
    except Exception:
        logger.exception('Device probe subprocess raised in the parent.')

    return [CPU]


def visibility_environment(
    selected: Sequence[str],
    available: Sequence[str] = (),
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the environment entries that restrict a child to *selected*.

    Empty when nothing is selected, which leaves the child seeing whatever this
    process does.

    *available* is every identifier the probe reported. A family that appears
    there and is named by nothing in *selected* is masked to empty, so that
    selecting the CPU hides the accelerators rather than silently leaving them
    in reach. Only families that were actually probed are touched: writing a
    mask for a runtime this machine does not have risks hiding the one it does,
    since HIP reads CUDA's variable as well as its own.

    An identifier this process cannot reach contributes nothing rather than
    raising -- a selection outlives the hardware it was made against.
    """
    if not selected:
        return {}

    environ = os.environ if environ is None else environ
    entries: dict[str, list[str]] = {
        variable: []
        for variable in (_maskable_family(i) for i in (*available, *selected))
        if variable is not None
    }

    for identifier in selected:
        variable = _maskable_family(identifier)

        if variable is None:
            continue

        _, _, ordinal = identifier.partition(':')
        entry = _resolve_through_inherited_mask(ordinal, environ.get(variable, ''))

        if entry is None:
            logger.warning(
                f'Device "{identifier}" is not visible to this process; '
                'it is dropped from the selection.'
            )
            continue

        entries[variable].append(entry)

    return {variable: ','.join(values) for variable, values in entries.items()}


def _maskable_family(identifier: str) -> str | None:
    """Return the variable that hides or reveals *identifier*, if anything does.

    torch exposes ROCm through its CUDA interface, so the two present
    identically and are told apart by the build rather than by the API; the
    probe resolves that once and spells the difference into the identifier.
    Nothing hides a CPU, which is why it maps to no variable at all.
    """
    variables: Final[Mapping[str, str]] = {
        'cuda': 'CUDA_VISIBLE_DEVICES',
        'rocm': 'HIP_VISIBLE_DEVICES',
        'xpu': 'ZE_AFFINITY_MASK',
    }
    family, _, _ = identifier.partition(':')
    return variables.get(family)


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
