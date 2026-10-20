"""Child-side subprocess entry points for the PtyChi backend.

Every function in this module runs INSIDE a spawned subprocess and touches the
GPU. The reconstruction loop itself lives in :mod:`.task`, which is where
``ptychi.api.task.PtychographyTask`` gets instantiated — that's the step that
acquires a CUDA context. Everything else (option translation, settings reads)
already ran parent-side; the child receives a finished
:class:`PtychographyTaskOptions` under the payload's ``task`` key, derives
pty-chi's task-data arrays from the ``ReconstructInput`` that travels with it,
and streams outputs back.

Which device pty-chi runs on is settled here rather than parent-side, because
this is the first process that can see one. ptychodus restricts the child to
the devices a run selected, so asking torch what survived that restriction is
the same question as asking what the user chose -- and a selection naming only
the CPU leaves nothing for ``Devices.GPU`` to mean.
"""

from __future__ import annotations

import logging
import pickle
from multiprocessing.queues import Queue
from typing import Any

from ...api.reconstruct import TAG_OUTPUT, SubprocessPayload
from .task import reconstruct_with_ptychi

logger = logging.getLogger(__name__)


def _resolve_default_device(options: Any) -> None:
    """Point the options at a GPU if one is visible here, and at the CPU if not."""
    import torch

    from ptychi.api.enums import Devices

    visible = (
        torch.cuda.is_available()
        or getattr(torch, 'xpu', None) is not None
        and (torch.xpu.is_available())
    )
    device = Devices.GPU if visible else Devices.CPU
    options.reconstructor_options.default_device = device
    logger.info(f'Reconstructing on the {device.name}.')


def run_reconstruct(payload: SubprocessPayload, queue: Queue[Any]) -> None:
    """Child entry point. Acquire a GPU context via PtychographyTask, stream outputs."""
    task_options = payload.options['task']
    _resolve_default_device(task_options)

    for output in reconstruct_with_ptychi(
        payload.require_input(), task_options, payload.options['num_sync_epochs']
    ):
        queue.put((TAG_OUTPUT, pickle.dumps(output)))
