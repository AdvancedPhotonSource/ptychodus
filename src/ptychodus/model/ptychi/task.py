"""The pty-chi task boundary: options serialization plus the task run loop.

This module is PARENT-SAFE TO IMPORT. Like :mod:`._payload` it pulls
``ptychi.api`` in for the options dataclasses, which imports torch for its type
annotations but acquires no CUDA context. :func:`reconstruct_with_ptychi` must
only be CALLED from a child process: it is the only place in the ptychodus tree
that instantiates ``ptychi.api.task.PtychographyTask``, and the import of that
class is deferred to call time so the module itself stays context-free.

Wire format
-----------

:func:`dump_task_options` emits a JSON envelope::

    {"reconstructor": "<enum-value>", "options": <options.get_dict()>}

The ``reconstructor`` field carries the pty-chi ``Reconstructors`` enum value
that :meth:`get_reconstructor_type` returns for these options. Envelope + token
together let :func:`load_task_options` pick the right algorithm-specific
subclass (``DMOptions``, ``LSQMLOptions``, ...) without a side-channel argument.

Without the envelope, PIE, ePIE and rPIE would serialize to identical
dictionaries -- ``EPIEReconstructorOptions`` and ``RPIEReconstructorOptions``
add no fields over ``PIEReconstructorOptions`` -- and could swap for each other
silently, running a different algorithm. The envelope closes that hazard: the
token is what selects the subclass, and it comes from the options object at
serialization time.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import numpy

from ptychi.api import (
    AutodiffPtychographyOptions,
    BHOptions,
    DMOptions,
    EPIEOptions,
    LSQMLOptions,
    PIEOptions,
    RAAROptions,
    RPIEOptions,
    Reconstructors,
)
from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.object import Object, ObjectPosition
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput
from ptychodus.api.typing import RealArrayType

logger = logging.getLogger(__name__)

__all__ = [
    'dump_task_options',
    'load_task_options',
    'reconstruct_with_ptychi',
]


_TASK_OPTIONS_CLS_BY_RECONSTRUCTOR: dict[Reconstructors, type[PtychographyTaskOptions]] = {
    Reconstructors.DM: DMOptions,
    Reconstructors.RAAR: RAAROptions,
    Reconstructors.PIE: PIEOptions,
    Reconstructors.EPIE: EPIEOptions,
    Reconstructors.RPIE: RPIEOptions,
    Reconstructors.LSQML: LSQMLOptions,
    Reconstructors.AD_PTYCHO: AutodiffPtychographyOptions,
    Reconstructors.BH: BHOptions,
}


def dump_task_options(options: PtychographyTaskOptions) -> str:
    """Serialize a fully-built options object for transport to a child process.

    The envelope carries the reconstructor token so PIE/ePIE/rPIE (which share
    a serialized field set) survive the round-trip as the correct subclass.
    """
    reconstructor = options.reconstructor_options.get_reconstructor_type()
    return json.dumps({'reconstructor': reconstructor.value, 'options': options.get_dict()})


def load_task_options(text: str) -> PtychographyTaskOptions:
    """Rebuild the options object that :func:`dump_task_options` serialized.

    The reconstructor token in the envelope selects the algorithm-specific
    subclass; ``load_from_dict`` then resolves nested options through the
    subclass's declared field annotations so no algorithm-specific field is
    silently dropped.
    """
    envelope = json.loads(text)

    if not isinstance(envelope, dict):
        raise ValueError(f'Expected a JSON object envelope; got {type(envelope).__name__}!')

    try:
        reconstructor = Reconstructors(envelope['reconstructor'])
    except KeyError as exc:
        raise ValueError(f'Options envelope is missing "{exc.args[0]}"!') from None
    except ValueError:
        raise ValueError(f'Unknown pty-chi reconstructor "{envelope["reconstructor"]}"!') from None

    try:
        task_options_cls = _TASK_OPTIONS_CLS_BY_RECONSTRUCTOR[reconstructor]
    except KeyError:
        raise ValueError(f'Unknown pty-chi reconstructor "{reconstructor}"!') from None

    options_dict = envelope.get('options')
    if not isinstance(options_dict, dict):
        raise ValueError('Options envelope "options" field must be a JSON object!')

    options = task_options_cls()
    options.load_from_dict(options_dict)
    return options


def _initial_opr_mode_weights(probe: ProbeSequence) -> RealArrayType:
    """Return the probe's OPR weights, or all weight on the primary mode."""
    try:
        return probe.get_opr_weights()
    except ValueError:
        pass

    initial_weights = numpy.zeros((probe.num_coherent_modes))
    initial_weights[0] = 1.0
    return initial_weights


def reconstruct_with_ptychi(
    parameters: ReconstructInput,
    task_options: PtychographyTaskOptions,
    num_sync_epochs: int,
) -> Iterator[ReconstructOutput]:
    """Instantiate ``PtychographyTask`` and yield a ``ReconstructOutput`` every
    ``num_sync_epochs`` epochs. The ``PtychographyTask`` import is deferred to
    call time so importing this module parent-side does not acquire a GPU
    context."""
    from ptychi.api.task import PtychographyTask

    num_epochs = task_options.reconstructor_options.num_epochs
    product_in = parameters.product
    object_in = product_in.object_
    object_geometry = object_in.get_geometry()

    # pty-chi stores probe positions in object pixel units; ptychodus stores
    # them in meters. The output path below applies the inverse mapping.
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in product_in.probe_positions:
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        position_x_px.append(object_point.coordinate_x_px)
        position_y_px.append(object_point.coordinate_y_px)

    # Task data goes in as keyword arguments; passing it through the *Options
    # objects still works but is deprecated and warns per field.
    task = PtychographyTask(
        task_options,
        diffraction_data=parameters.diffraction_patterns,
        object_data=object_in.get_array(),
        probe_data=product_in.probes.get_array(),
        probe_position_x_px=numpy.array(position_x_px),
        probe_position_y_px=numpy.array(position_y_px),
        opr_mode_weights_data=_initial_opr_mode_weights(product_in.probes),
        valid_pixel_mask=numpy.logical_not(parameters.bad_pixels),
    )

    with task:
        epoch = 0

        task_reconstructor = task.reconstructor

        if task_reconstructor is None:
            raise RuntimeError('Task reconstructor is None!')

        loss_tracker = task_reconstructor.loss_tracker

        while epoch < num_epochs:
            step_epochs = min(num_sync_epochs, num_epochs - epoch)
            task.run(step_epochs)

            losses: list[LossValue] = list()
            epoch_array = loss_tracker.table['epoch'].to_numpy()
            loss_array = loss_tracker.table['loss'].to_numpy()

            for e, loss in zip(epoch_array.flat, loss_array.flat):
                losses.append(LossValue(epoch=e, value=loss.item()))

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

            corrected_position_x_px = task.get_probe_positions_x(as_numpy=True)
            corrected_position_y_px = task.get_probe_positions_y(as_numpy=True)
            corrected_scan_points: list[ProbePosition] = list()

            for uncorrected_point, pos_x_px, pos_y_px in zip(
                product_in.probe_positions, corrected_position_x_px, corrected_position_y_px
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

            yield ReconstructOutput(product=product, progress=epoch)
