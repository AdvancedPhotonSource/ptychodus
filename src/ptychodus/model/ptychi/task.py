"""The pty-chi task boundary: options serialization plus the task run loop.

This module is PARENT-SAFE TO IMPORT. Like :mod:`._payload` it pulls
``ptychi.api`` in for the options dataclasses, which imports torch for its type
annotations but acquires no CUDA context. :func:`reconstruct_with_ptychi` must
only be CALLED from a child process: it is the only place in the ptychodus tree
that instantiates ``ptychi.api.task.PtychographyTask``, and the import of that
class is deferred to call time so the module itself stays context-free.

Wire format
-----------

:func:`dump_task_options` emits a bare ``json.dumps(options.get_dict())`` with
no envelope, matching the ``settings.json`` pattern documented in pty-chi's own
``using_pty_chi/io.rst``. ``get_dict()`` drops the large array fields, which
costs nothing here: ptychodus never puts data into an options object, because
pty-chi takes task data as ``PtychographyTask`` keyword arguments instead.

The blob does not record which algorithm produced it, and it cannot:
``get_reconstructor_type()`` is a method, so it never lands in the dict.
:func:`load_task_options` therefore takes the reconstructor as an argument.

Two classes with different field sets do not silently mix -- ``load_from_dict``
raises on the first unrecognized key -- but ``EPIEReconstructorOptions`` and
``RPIEReconstructorOptions`` add no fields over ``PIEReconstructorOptions``, so
PIE, ePIE and rPIE serialize to identical dictionaries and swap for each other
without a murmur, quietly running a different algorithm. Callers must therefore
derive the value from the options object with :func:`reconstructor_argument`
rather than hand-writing the token.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping
from typing import Final

import numpy

from ptychi.api import (
    AutodiffPtychographyOptions,
    BHOptions,
    DMOptions,
    EPIEOptions,
    LSQMLOptions,
    PIEOptions,
    RAAROptions,
    Reconstructors,
    RPIEOptions,
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
    'OPTIONS_CLASSES',
    'RECONSTRUCTOR_CHOICES',
    'dump_task_options',
    'load_task_options',
    'reconstruct_with_ptychi',
    'reconstructor_argument',
]


# The task-options classes ptychodus builds, one per entry in the algorithm
# table in :mod:`.reconstructor`. ``AutodiffOptions`` (``AD_GENERAL``) is
# deliberately absent: ``ptychi/api/options/__init__.py`` does not import
# ``ad_general``, so it is not reachable as ``ptychi.api.AutodiffOptions``.
_TASK_OPTIONS_CLASSES: Final[tuple[type[PtychographyTaskOptions], ...]] = (
    DMOptions,
    RAAROptions,
    PIEOptions,
    EPIEOptions,
    RPIEOptions,
    LSQMLOptions,
    AutodiffPtychographyOptions,
    BHOptions,
)


def _build_options_classes() -> dict[Reconstructors, type[PtychographyTaskOptions]]:
    """Invert pty-chi's ``get_reconstructor_type()`` into a lookup table.

    pty-chi maps the reconstructor enum to the reconstructor *implementation*
    class (``ptychi.maps``) but provides no map to the task-options class, so
    ptychodus builds that half here. Deriving the keys from pty-chi rather than
    hardcoding them keeps this vocabulary from drifting; only the class tuple
    above is hand-maintained.
    """
    classes: dict[Reconstructors, type[PtychographyTaskOptions]] = dict()

    for options_class in _TASK_OPTIONS_CLASSES:
        reconstructor = options_class().reconstructor_options.get_reconstructor_type()
        collision = classes.get(reconstructor)

        if collision is not None:
            raise RuntimeError(
                f'pty-chi reports reconstructor type "{reconstructor}" for both'
                f' {collision.__name__} and {options_class.__name__}!'
            )

        classes[reconstructor] = options_class

    return classes


OPTIONS_CLASSES: Final[Mapping[Reconstructors, type[PtychographyTaskOptions]]] = (
    _build_options_classes()
)
"""Reconstructor type to task-options class."""

RECONSTRUCTOR_CHOICES: Final[tuple[str, ...]] = tuple(
    reconstructor.value for reconstructor in OPTIONS_CLASSES
)
"""The accepted reconstructor tokens, for a command-line ``choices`` list."""


def dump_task_options(options: PtychographyTaskOptions) -> str:
    """Serialize a fully-built options object for transport to a child process."""
    return json.dumps(options.get_dict())


def reconstructor_argument(options: PtychographyTaskOptions) -> str:
    """Return the reconstructor token that matches these options.

    Launchers must derive the token from the options object rather than
    hand-writing it; see the module docstring.
    """
    return str(options.reconstructor_options.get_reconstructor_type().value)


def load_task_options(text: str, reconstructor: Reconstructors) -> PtychographyTaskOptions:
    """Rebuild the options object that :func:`dump_task_options` serialized.

    ``reconstructor`` selects the algorithm-specific subclass. It is required
    because ``load_from_dict`` resolves nested options through declared field
    annotations, so loading into a plain ``PtychographyTaskOptions`` would
    downcast ``reconstructor_options`` to its base class and silently drop
    every algorithm-specific field.
    """
    try:
        options_class = OPTIONS_CLASSES[reconstructor]
    except KeyError:
        raise ValueError(f'Unknown pty-chi reconstructor "{reconstructor}"!') from None

    contents = json.loads(text)

    if not isinstance(contents, dict):
        raise ValueError(
            f'Expected a JSON object of pty-chi options; got {type(contents).__name__}!'
        )

    options = options_class()
    options.load_from_dict(contents)

    loaded_reconstructor = options.reconstructor_options.get_reconstructor_type()

    if loaded_reconstructor != reconstructor:
        raise RuntimeError(
            f'{options_class.__name__} reports reconstructor type'
            f' "{loaded_reconstructor}" but "{reconstructor}" was requested!'
        )

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
