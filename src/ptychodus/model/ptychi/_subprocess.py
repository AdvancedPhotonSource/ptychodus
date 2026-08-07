"""Child-side subprocess entry points for the PtyChi backend.

Every function in this module runs INSIDE a spawned subprocess and touches
the GPU. This is the only place in the ptychodus tree that instantiates
``ptychi.api.task.PtychographyTask`` — that's the step that acquires a CUDA
context. Everything else (option translation, settings reads) already ran
parent-side; the child receives a finished :class:`PtychographyTaskOptions`
and streams outputs back.

The device-probing entry point is a one-shot: it enumerates devices and
exits. Parent-side callers spawn it via :mod:`.device` when populating
:class:`PtyChiDeviceRepository`.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Iterator
from multiprocessing.queues import Queue
from typing import Any

import numpy

from ptychodus.api.object import Object, ObjectPosition
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput

from ..processing.subprocess_reconstructor import TAG_OUTPUT
from ._payload import PtyChiPayload

logger = logging.getLogger(__name__)


def reconstruct_with_ptychi(
    parameters: ReconstructInput,
    payload: PtyChiPayload,
) -> Iterator[ReconstructOutput]:
    """Instantiate ``PtychographyTask`` and yield a ``ReconstructOutput`` every
    ``num_sync_epochs`` epochs. The ``PtychographyTask`` import is deferred to
    call time so parent-side test collection of this module (if it ever
    happens) does not acquire a GPU context."""
    from ptychi.api.task import PtychographyTask

    task_options = payload.task_options
    num_sync_epochs = payload.num_sync_epochs
    num_epochs = task_options.reconstructor_options.num_epochs
    task = PtychographyTask(task_options)

    with task:
        epoch = 0
        step_epochs = num_sync_epochs

        task_reconstructor = task.reconstructor

        if task_reconstructor is None:
            raise RuntimeError('Task reconstructor is None!')

        loss_tracker = task_reconstructor.loss_tracker

        while epoch < num_epochs:
            task.run(step_epochs)

            losses: list[LossValue] = list()
            epoch_array = loss_tracker.table['epoch'].to_numpy()
            loss_array = loss_tracker.table['loss'].to_numpy()

            for e, loss in zip(epoch_array.flat, loss_array.flat):
                losses.append(LossValue(epoch=e, value=loss.item()))

            product_in = parameters.product
            object_in = product_in.object_
            object_out = Object(
                array=numpy.array(task.get_data_to_cpu('object', as_numpy=True)),
                layer_spacing_m=object_in.layer_spacing_m,
                pixel_geometry=object_in.get_pixel_geometry(),
                center=object_in.get_center(),
            )
            probe_out = ProbeSequence(
                array=numpy.array(task.get_data_to_cpu('probe', as_numpy=True)),
                opr_weights=numpy.array(task.get_data_to_cpu('opr_mode_weights', as_numpy=True)),
                pixel_geometry=product_in.probes.get_pixel_geometry(),
            )

            position_x_px = task.get_probe_positions_x(as_numpy=True)
            position_y_px = task.get_probe_positions_y(as_numpy=True)
            object_geometry = object_in.get_geometry()
            corrected_scan_points: list[ProbePosition] = list()

            for uncorrected_point, pos_x_px, pos_y_px in zip(
                product_in.probe_positions, position_x_px, position_y_px
            ):
                object_point = ObjectPosition(
                    index=uncorrected_point.index,
                    coordinate_x_px=float(pos_x_px),
                    coordinate_y_px=float(pos_y_px),
                )
                scan_point = object_geometry.map_coordinates_object_to_probe(object_point)
                corrected_scan_points.append(scan_point)

            product = Product(
                metadata=product_in.metadata,
                probe_positions=ProbePositionSequence(corrected_scan_points),
                probes=probe_out,
                object_=object_out,
                losses=losses,
            )

            epoch += step_epochs
            step_epochs = min(step_epochs, num_epochs - epoch)

            yield ReconstructOutput(product=product, progress=epoch)


def run_reconstruct(payload: PtyChiPayload, queue: Queue[Any]) -> None:
    """Child entry point. Acquire a GPU context via PtychographyTask, stream outputs."""
    for output in reconstruct_with_ptychi(payload.reconstruct_input, payload):
        queue.put((TAG_OUTPUT, pickle.dumps(output)))


def probe_device_list() -> list[str]:
    """One-shot device enumeration for the parent-side device repository."""
    import ptychi

    return [f'{d.name} ({d.torch_device})' for d in ptychi.list_available_devices()]


def probe_devices(_payload: Any, queue: Queue[Any]) -> None:
    """Spawn-safe entry point that emits the device list on the queue and exits."""
    queue.put((TAG_OUTPUT, pickle.dumps(probe_device_list())))
