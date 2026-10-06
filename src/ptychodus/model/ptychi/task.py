"""The pty-chi task boundary: options serialization plus the task run loop.

This module is PARENT-SAFE TO IMPORT. Like :mod:`._payload` it pulls
``ptychi.api`` in for the options dataclasses, which imports torch for its type
annotations but acquires no CUDA context. :func:`reconstruct_with_ptychi` must
only be CALLED from a child process: it is the only place in the ptychodus tree
that instantiates ``ptychi.api.task.PtychographyTask``, and the import of that
class is deferred to call time so the module itself stays context-free.

Wire format
-----------

:func:`dump_task_options` emits ``options.get_dict()`` as a bare JSON object.
pty-chi >= 2.1.0 stamps every options dict -- the top-level one and each nested
one -- with its own ``options_class_name``, so the payload names the class that
wrote it and :func:`load_task_options` can pick the right algorithm-specific
subclass (``DMOptions``, ``LSQMLOptions``, ...) without a side-channel argument.

That stamp is what keeps PIE, ePIE and rPIE apart. Their dicts are otherwise
identical -- ``EPIEReconstructorOptions`` and ``RPIEReconstructorOptions`` add
no fields over ``PIEReconstructorOptions`` -- so before pty-chi 2.1.0 the three
could swap for each other silently and run a different algorithm. Ptychodus
carried its own ``{"reconstructor": ..., "options": ...}`` envelope to close
that hazard; the upstream class name supersedes it.

:meth:`load_from_dict` is called with ``strict=True``, which checks the stamp
of every nested options object against the field it is being loaded into, so a
payload assembled from mismatched parts raises instead of loading quietly.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections.abc import Iterator, Sequence
from typing import Any

import numpy

import ptychi.api
from ptychi.api import ObjectPosOriginCoordsMethods
from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectPosition
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import (
    ReconstructInput,
    ReconstructOutput,
    warn_if_propagation_regime_disagrees,
)
from ptychodus.api.typing import RealArrayType

logger = logging.getLogger(__name__)

__all__ = [
    'align_task_options_with_product',
    'dump_task_options',
    'load_task_options',
    'reconstruct_with_ptychi',
]


def dump_task_options(options: PtychographyTaskOptions) -> str:
    """Serialize a fully-built options object for transport to a child process.

    ``get_dict`` stamps the dict with ``options_class_name``, which is what lets
    PIE/ePIE/rPIE (which share a serialized field set) survive the round-trip as
    the correct subclass.
    """
    return json.dumps(options.get_dict())


def load_task_options(text: str) -> PtychographyTaskOptions:
    """Rebuild the options object that :func:`dump_task_options` serialized.

    ``options_class_name`` selects the algorithm-specific subclass.
    ``load_from_dict`` then resolves nested options through that subclass's
    declared field annotations, so no algorithm-specific field is silently
    dropped, and ``strict=True`` rejects a payload whose nested stamps disagree
    with the fields they are being loaded into.
    """
    options_dict = json.loads(text)

    if not isinstance(options_dict, dict):
        raise ValueError(f'Expected a JSON object; got {type(options_dict).__name__}!')

    try:
        options_class_name = options_dict['options_class_name']
    except KeyError:
        raise ValueError('Options are missing "options_class_name"!') from None

    # pty-chi stamps the class name but offers no dict-to-options factory of its
    # own, so resolve it here. Going through ``ptychi.api``, which exports every
    # ``PtychographyTaskOptions`` subclass, means a pty-chi release that adds an
    # algorithm needs no change on this side, and the subclass test rejects both
    # an unknown name and an arbitrary attribute that happens to match one.
    options_cls = getattr(ptychi.api, options_class_name, None)

    if not isinstance(options_cls, type) or not issubclass(options_cls, PtychographyTaskOptions):
        raise ValueError(f'Unknown pty-chi options class "{options_class_name}"!')

    options = options_cls()
    options.load_from_dict(options_dict, strict=True)
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


def _overwrite(options: Any, name: str, value: Any) -> None:
    """Write `value` onto `options.name`, logging what changed.

    Warns only when the caller had expressed an intent that disagrees, i.e. the
    old value was neither pty-chi's default nor the value being written.
    """
    options_cls = type(options)
    old = getattr(options, name)
    setattr(options, name, value)
    # Read back rather than reusing `value`: pty-chi's mode='before' validators
    # normalize array-likes to lists, so both the log line and the comparison
    # below see the stored form and an ndarray does not warn against itself.
    new = getattr(options, name)
    logger.debug('%s.%s: %r -> %r', options_cls.__name__, name, old, new)

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


def _align_probe_pixel_geometry(
    aligned: PtychographyTaskOptions,
    product: Product,
    object_geometry: PixelGeometry,
    rel_tol: float = 1e-9,
) -> None:
    """Describe the product's probe sampling on `aligned.probe_options`.

    pty-chi inherits the object value for each probe pixel field left at
    ``None``, and it decides whether to resample by comparing pixel sizes rather
    than by testing for ``None``. Writing ``None`` when the two agree therefore
    both states the intent and keeps the no-resampling fast path.

    Width and aspect ratio are inherited independently upstream, so they are
    tested independently here.

    Pixel sizes agreeing to within `rel_tol` name the same sampling. A product
    round-tripped through HDF5 stores the probe and object geometries
    independently, so an exact comparison would let float noise engage pty-chi's
    Fourier resampling for what is physically an unresampled grid.
    """
    try:
        probe_geometry = product.probes.get_pixel_geometry()
    except ValueError:
        logger.debug('Product carries no probe pixel geometry; probe sampling follows the object.')
        return

    probe_options = aligned.probe_options

    probe_width_m = probe_geometry.width_m
    _overwrite(
        probe_options,
        'pixel_size_m',
        None
        if math.isclose(probe_width_m, object_geometry.width_m, rel_tol=rel_tol)
        else probe_width_m,
    )

    probe_aspect_ratio = probe_geometry.get_aspect_ratio()
    _overwrite(
        probe_options,
        'pixel_size_aspect_ratio',
        None
        if math.isclose(probe_aspect_ratio, object_geometry.get_aspect_ratio(), rel_tol=rel_tol)
        else probe_aspect_ratio,
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

    `free_space_propagation_distance_m` follows the product's declared propagation
    regime: infinite for far field, and the demagnified detector distance for near
    field. A caller-supplied distance that disagrees is overwritten with a warning,
    as any other product-derived field would be. A product whose declared regime
    contradicts its own geometry warns separately and is not corrected.

    The probe pixel fields are written from the product's probe pixel geometry
    by :func:`_align_probe_pixel_geometry`, which leaves them ``None`` -- pty-chi
    then inherits the object values -- whenever the two samplings agree. A
    product whose probe carries no pixel geometry leaves them untouched.

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

    # Near field: pty-chi propagates in the equivalent parallel-beam geometry, so a
    # cone beam contributes its demagnified distance. Without a focusing optic the
    # magnification is 1 and this is the detector distance unchanged.
    _overwrite(
        aligned.data_options,
        'free_space_propagation_distance_m',
        math.inf if metadata.far_field else metadata.detector_distance_m / metadata.magnification,
    )

    warn_if_propagation_regime_disagrees(product)

    _overwrite(aligned.probe_options.power_constraint, 'probe_power', metadata.probe_photon_count)

    _align_probe_pixel_geometry(aligned, product, object_geometry)

    return aligned


def _recover_layer_spacing_m(task: Any, object_in: Object) -> Sequence[float]:
    """Return the slice spacings the task ended with, falling back to the ones it started with.

    pty-chi can optimize slice spacings, but its readback API covers only the object,
    probe, positions and OPR weights, so the optimized values are reachable only through
    the task's own object. That is private structure and has churned before, so every
    failure here degrades to the input spacing rather than propagating: a stale thickness
    is a cosmetic loss, whereas a wrong-length one would make the Object constructor
    raise.
    """
    num_spacings = object_in.num_layers - 1

    if num_spacings < 1:
        return list(object_in.layer_spacing_m)

    try:
        spacing_m = [float(value) for value in task.object.slice_spacings.data.detach().cpu()]
    except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
        logger.warning(
            'Could not read slice spacings back from pty-chi (%s); keeping the input '
            'spacing. Any slice-spacing optimization is not reflected in the output.',
            exc,
        )
        return list(object_in.layer_spacing_m)

    if len(spacing_m) != num_spacings:
        logger.warning(
            'pty-chi returned %d slice spacing(s) for a %d-layer object; keeping the input '
            'spacing.',
            len(spacing_m),
            object_in.num_layers,
        )
        return list(object_in.layer_spacing_m)

    return spacing_m


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

    # A hand-built options object that skipped alignment. pty-chi neither validates
    # NaN nor warns -- both of its near-field guards test `< inf`, which NaN fails --
    # so the only other symptom would be a NaN loss on the first epoch.
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
    position_x_px = object_geometry.map_probe_positions_to_object_x_px(product_in.probe_positions)
    position_y_px = object_geometry.map_probe_positions_to_object_y_px(product_in.probe_positions)

    # Task data goes in as keyword arguments; passing it through the *Options
    # objects still works but is deprecated and warns per field.
    task = PtychographyTask(
        task_options,
        diffraction_data=parameters.diffraction_patterns,
        object_data=object_in.get_array(),
        probe_data=product_in.probes.get_array(),
        probe_position_x_px=position_x_px,
        probe_position_y_px=position_y_px,
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
                layer_spacing_m=_recover_layer_spacing_m(task, object_in),
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
