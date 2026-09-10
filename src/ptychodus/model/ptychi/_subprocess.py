"""Child-side subprocess entry points for the PtyChi backend.

Every function in this module runs INSIDE a spawned subprocess and touches the
GPU. The reconstruction loop itself lives in :mod:`.task`, which is where
``ptychi.api.task.PtychographyTask`` gets instantiated — that's the step that
acquires a CUDA context. Everything else (option translation, settings reads)
already ran parent-side; the child receives a finished
:class:`PtychographyTaskOptions`, derives pty-chi's task-data arrays from the
``ReconstructInput`` that travels with it, and streams outputs back.

The device-probing entry point is a one-shot: it enumerates devices and
exits. Parent-side callers spawn it via :mod:`.device` when populating
:class:`PtyChiDeviceRepository`.
"""

from __future__ import annotations

import logging
import pickle
from multiprocessing.queues import Queue
from typing import Any

from ..processing.subprocess_reconstructor import TAG_OUTPUT
from ._payload import PtyChiPayload
from .task import reconstruct_with_ptychi

logger = logging.getLogger(__name__)


def run_reconstruct(payload: PtyChiPayload, queue: Queue[Any]) -> None:
    """Child entry point. Acquire a GPU context via PtychographyTask, stream outputs."""
    for output in reconstruct_with_ptychi(
        payload.reconstruct_input, payload.task_options, payload.num_sync_epochs
    ):
        queue.put((TAG_OUTPUT, pickle.dumps(output)))


def probe_device_list() -> list[str]:
    """One-shot device enumeration for the parent-side device repository."""
    import ptychi

    return [f'{d.name} ({d.torch_device})' for d in ptychi.list_available_devices()]


def probe_devices(_payload: Any, queue: Queue[Any]) -> None:
    """Spawn-safe entry point that emits the device list on the queue and exits."""
    queue.put((TAG_OUTPUT, pickle.dumps(probe_device_list())))
