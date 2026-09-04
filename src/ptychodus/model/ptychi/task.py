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

import copy
import json
import logging
import math
from collections.abc import Iterator
from typing import Any

import numpy

from ptychi.api import (
    AutodiffPtychographyOptions,
    BHOptions,
    DMOptions,
    EPIEOptions,
    LSQMLOptions,
    ObjectPosOriginCoordsMethods,
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
    'align_task_options_with_product',
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


def _differs(lhs: Any, rhs: Any) -> bool:
    """Compare two option values, reading an equal list and tuple as the same.

    ``position_origin_coords`` and ``slice_spacings_m`` are typed ``list |
    tuple``, so a bare ``!=`` would call ``[0.0, 0.0]`` and ``(0.0, 0.0)`` a
    change. Numbers compare exactly: the question is whether the caller supplied
    a different value, and a tolerance would only hide genuine disagreement
    between two products.
    """
    if isinstance(lhs, (list, tuple)) and isinstance(rhs, (list, tuple)):
        return list(lhs) != list(rhs)

    return bool(lhs != rhs)


def _overwrite(options: Any, name: str, value: Any, *, unset: bool = False) -> None:
    """Write `value` onto `options.name`, logging what changed.

    Warns only when the caller had expressed an intent that disagrees, i.e. the
    old value was neither pty-chi's default nor the value being written. Pass
    `unset` for a field whose incoming value is an internal placeholder rather
    than a caller's intent; it suppresses the warning but not the debug line.
    """
    options_cls = type(options)
    old = getattr(options, name)
    setattr(options, name, value)
    # Read back rather than reusing `value`: pty-chi's mode='before' validators
    # normalize array-likes to lists, so both the log line and the comparison
    # below see the stored form and an ndarray does not warn against itself.
    new = getattr(options, name)
    logger.debug('%s.%s: %r -> %r', options_cls.__name__, name, old, new)

    if unset:
        return

    # Defaults come off a throwaway instance rather than ``dataclasses.fields``:
    # pty-chi declares its constrained scalars with ``PydanticField(...)``, whose
    # ``.default`` is a ``FieldInfo`` and not the value -- and it does so
    # inconsistently, so the field route cannot be used uniformly. Construction
    # costs ~100 us against the eight writes one alignment performs.
    default = getattr(options_cls(), name)

    if _differs(old, default) and _differs(old, new):
        logger.warning(
            '%s.%s was %r but the product requires %r; overwriting.',
            options_cls.__name__,
            name,
            old,
            new,
        )


def align_task_options_with_product(
    task_options: PtychographyTaskOptions, product: Product
) -> PtychographyTaskOptions:
    """Return a copy of `task_options` whose product-derived fields agree with `product`.

    This is the single source of truth for which pty-chi options describe the
    ptychodus product rather than user settings. Everything it writes is either
    a value read off `product` or the position-origin convention that
    :func:`reconstruct_with_ptychi` relies on when it maps probe positions to
    object pixels.

    `task_options` is not modified, so a caller can compare before against
    after::

        aligned = align_task_options_with_product(options, product)
        before, after = dump_task_options(options), dump_task_options(aligned)

    ``get_dict`` strips the task-data arrays, so that pair is a readable diff of
    the settings alone. The copy is deep: a caller that has attached task arrays
    to the options pays to duplicate them, though no in-ptychodus path does.

    `free_space_propagation_distance_m` is the one field split between a setting
    and the product. An infinite incoming value means the caller chose far-field
    propagation and is left alone; anything else is near-field, and the product
    supplies the distance.

    Per-field pydantic validation runs on each assignment, so a degenerate
    product (`pixel_size_m` must be > 0, `probe_power` >= 0) raises
    ``ValidationError`` from here rather than from the options constructor.
    """
    aligned = copy.deepcopy(task_options)
    metadata = product.metadata
    object_geometry = product.object_.get_pixel_geometry()

    object_options = aligned.object_options
    _overwrite(object_options, 'pixel_size_m', object_geometry.width_m)
    _overwrite(object_options, 'pixel_size_aspect_ratio', object_geometry.get_aspect_ratio())

    # pty-chi types slice spacings as a serializable list, not an ndarray. Test
    # length rather than truthiness: a product read back from HDF5 carries an
    # ndarray, and `if array` raises for every length but one.
    slice_spacings_m = product.object_.layer_spacing_m
    _overwrite(
        object_options,
        'slice_spacings_m',
        [float(spacing) for spacing in slice_spacings_m] if len(slice_spacings_m) > 0 else None,
    )

    # ptychodus hands pty-chi probe positions already mapped to 0-based object
    # pixel indices (see `map_coordinates_probe_to_object` below), and pty-chi
    # computes `positions_pxind = positions + position_origin_coords`. A zero
    # origin is what makes those two agree; the default of SUPPORT would add
    # half the object shape to every position.
    _overwrite(
        object_options,
        'determine_position_origin_coords_by',
        ObjectPosOriginCoordsMethods.SPECIFIED,
    )
    _overwrite(object_options, 'position_origin_coords', [0.0, 0.0])

    _overwrite(aligned.data_options, 'wavelength_m', metadata.probe_wavelength_m)

    propagation_distance_m = aligned.data_options.free_space_propagation_distance_m

    if not math.isinf(propagation_distance_m):
        _overwrite(
            aligned.data_options,
            'free_space_propagation_distance_m',
            metadata.detector_distance_m,
            unset=math.isnan(propagation_distance_m),
        )

    _overwrite(aligned.probe_options.power_constraint, 'probe_power', metadata.probe_photon_count)

    return aligned


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

    # The near-field placeholder that PtyChiCommon emits, escaping into a run.
    # pty-chi neither validates it nor warns -- both of its near-field guards
    # test `< inf`, which NaN fails -- so the only other symptom would be a NaN
    # loss on the first epoch.
    if math.isnan(task_options.data_options.free_space_propagation_distance_m):
        raise ValueError(
            'free_space_propagation_distance_m is NaN; call '
            'align_task_options_with_product() before reconstructing.'
        )

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
        position_x_px.append(object_point.x_px)
        position_y_px.append(object_point.y_px)

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
                    x_px=float(pos_x_px),
                    y_px=float(pos_y_px),
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
