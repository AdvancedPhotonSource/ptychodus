"""The child side of the batch-LamNI fan-out: one scan, on one device.

This module exists for its import-time emptiness. ``CUDA_VISIBLE_DEVICES`` only takes
effect if it is set before torch is first imported, and a spawned child imports whatever
its entry point's module imports in order to find the function. So nothing heavy may
appear at module scope here, and every real import is deferred until
:func:`reconstruct_scan` has set the environment.

The payload is plain data for the same reason. ``multiprocessing`` unpickles a child's
arguments during bootstrap, before the target runs, so a payload holding a live settings
object or a pty-chi options object would import those modules -- and torch with them --
while the environment still says nothing about which device to use. Text and numbers
cross instead, and this side rebuilds from them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing import Queue

__all__ = ['reconstruct_scan']

# Rank, as a distributed launcher reports it. A worker here is not a rank of anything: it
# reconstructs one whole scan by itself, into a directory no other worker writes. Left in
# place, an inherited SLURM_PROCID would convince every worker it was a non-main rank and
# suppress the writes that are the point of the run.
_RANK_ENVIRONMENT_VARIABLES = ('RANK', 'LOCAL_RANK', 'SLURM_PROCID', 'SLURM_LOCALID')


def _redirect_console(output_directory: Any) -> None:
    """Point this process's stdout and stderr at the scan's own log file.

    At the file-descriptor level, not by reassigning ``sys.stdout``: pty-chi's progress
    bars and torch's chatter come out of C extensions writing to fd 1 and fd 2 directly,
    which a Python-level reassignment never sees. Both descriptors go to one file so that
    a crash's stderr lands in order against the stdout that preceded it.
    """
    import sys

    from ptychodus.api.io import StandardFileLayout

    output_directory.mkdir(parents=True, exist_ok=True)

    with open(StandardFileLayout.LOG.path(output_directory), 'w') as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)

    sys.stdout = os.fdopen(os.dup(1), 'w', buffering=1)
    sys.stderr = os.fdopen(os.dup(2), 'w', buffering=1)


def reconstruct_scan(payload: dict[str, Any], queue: Queue[Any]) -> None:
    """Reconstruct the one scan this child was spawned for.

    Raising is how a failure is reported: the subprocess protocol marshals the exception
    back to the parent, which counts the scan as failed and, under
    ``misc.continue_on_error``, carries on with the rest.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = payload['cuda_visible_devices']

    for name in _RANK_ENVIRONMENT_VARIABLES:
        os.environ.pop(name, None)

    # Everything below may import torch, now that the device is chosen.
    import logging
    from pathlib import Path

    from ptychodus.api.diffraction import CropRegion
    from ptychodus.api.plugins import PluginRegistry
    from ptychodus.cli._batch_lamni_scan import ScanRequest, reconstruct_one_scan
    from ptychodus.cli._batch_lamni_settings import load_batch_settings_from_text
    from ptychodus.cli._reconstruct_common import install_signal_handlers, load_ptychi_options

    settings = load_batch_settings_from_text(payload['settings_text'])
    scan = payload['scan']
    output_directory = settings.output_directory(scan['scan_no'])

    _redirect_console(output_directory)

    # The forwarder to the parent is already on the root logger; this adds the same
    # records to the redirected stderr, so the scan's directory explains itself without
    # the parent's scrollback.
    logging.basicConfig(
        level=settings.misc.log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        force=False,
    )

    # A Ctrl-C reaches the whole process group, so this child gets its own SIGINT. Caught
    # here, it stops at the next sync point and still returns through the protocol;
    # uncaught, the child would die without its sentinel and leave the parent's reader
    # waiting on a queue nothing will ever close.
    cancellation = install_signal_handlers()

    region = payload['read_region']
    read_region = (
        None
        if region is None
        else CropRegion(x_range=tuple(region['x_range']), y_range=tuple(region['y_range']))
    )

    registry = PluginRegistry.load_plugins()
    bad_pixels = None
    bad_pixels_file = settings.diffraction.bad_pixels_file

    if bad_pixels_file is not None:
        reader = registry.bad_pixels_file_readers.get_strategy_by_name(
            settings.diffraction.bad_pixels_file_type
        )
        bad_pixels = reader.read(Path(bad_pixels_file))

    options = load_ptychi_options(None)

    if payload['ptychi_options_json'] is not None:
        from ptychodus.model.ptychi.task import load_task_options

        options = load_task_options(payload['ptychi_options_json'])

    completed = reconstruct_one_scan(
        settings,
        payload['settings_text'],
        registry,
        ScanRequest(
            scan_no=scan['scan_no'],
            label=scan['label'],
            encoder_angle_deg=scan['encoder_angle_deg'],
            text=scan['text'],
        ),
        options,
        read_region=read_region,
        bad_pixels=bad_pixels,
        cancellation=cancellation,
    )

    queue.put(('output', {'scan_no': scan['scan_no'], 'completed': completed}))
