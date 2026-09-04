"""Consistency checks between ptychodus pty-chi settings and pty-chi Options.

pty-chi's ``Options`` classes are Pydantic dataclasses that enforce every
``ge``/``gt``/``le`` constraint at construction time. These tests build the full
per-reconstructor task-options object from ptychodus settings and let pty-chi's
validators run, so any default or bound in ``ptychodus.model.ptychi.settings``
that pty-chi would reject surfaces here rather than as a reconstruction-time
crash. No GPU or real dataset is required — per-field validation runs at
construction; the task-level ``.check()`` and ``check_task_data()`` cross
validation only runs inside ``PtychographyTask``.

Task data no longer travels in the options at all: pty-chi takes it as
``PtychographyTask`` keyword arguments and deprecates the option fields, so
these tests also pin down that separation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import operator

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.parameters import IntegerParameter, RealParameter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.settings import SettingsRegistry
from ptychi.api import LSQMLOptions, ObjectPosOriginCoordsMethods, Reconstructors
from ptychi.api.options.data import PtychographyDataOptions

from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary

# Deliberately imported at module scope: this file already does
# ``pytest.importorskip('ptychi')`` above.
from ptychodus.model.ptychi.algorithms import (  # noqa: E402
    AutodiffAlgorithm,
    BHAlgorithm,
    DMAlgorithm,
    EPIEAlgorithm,
    LSQMLAlgorithm,
    PIEAlgorithm,
    PtyChiAlgorithm,
    PtyChiCommon,
    RAARAlgorithm,
    RPIEAlgorithm,
    build_algorithms,
)
from ptychodus.model.ptychi.task import (  # noqa: E402
    _initial_opr_mode_weights,
    align_task_options_with_product,
    dump_task_options,
    load_task_options,
)

PIXEL_M = 1.0e-9
# Every expected value below is a distinct witness: none coincides with a
# pty-chi default (1.0 m, 1e-9 m, inf, 0.0), so a test cannot pass by accident
# if the helper drops the field it is checking.
DETECTOR_DISTANCE_M = 2.5
PROBE_ENERGY_EV = 10_000.0
PROBE_PHOTON_COUNT = 1.25e6
OBJ_HEIGHT_PX = 32
OBJ_WIDTH_PX = 40
PROBE_HEIGHT_PX = 8
PROBE_WIDTH_PX = 8
NUM_PATTERNS = 3


def _make_reconstruct_input() -> ReconstructInput:
    rng = numpy.random.default_rng(0)
    obj = Object(
        array=(
            rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
            + 1j * rng.standard_normal((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
        ).astype(numpy.complex128),
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[],
    )
    probes = ProbeSequence(
        array=(
            rng.standard_normal((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
            + 1j * rng.standard_normal((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX))
        ).astype(numpy.complex128),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
    )
    positions = ProbePositionSequence(
        [ProbePosition(index=i, x_m=i * PIXEL_M, y_m=-i * PIXEL_M) for i in range(NUM_PATTERNS)]
    )
    product = Product(
        metadata=ProductMetadata(
            name='test',
            comments='',
            detector_distance_m=DETECTOR_DISTANCE_M,
            probe_energy_eV=PROBE_ENERGY_EV,
            probe_photon_count=PROBE_PHOTON_COUNT,
            exposure_time_s=1.0,
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=positions,
        probes=probes,
        object_=obj,
        losses=[],
    )
    patterns = rng.random((NUM_PATTERNS, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)).astype(numpy.float32)
    bad_pixels = numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_)
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=bad_pixels,
        product=product,
    )


def _make_library() -> PtyChiReconstructorLibrary:
    return PtyChiReconstructorLibrary(
        SettingsRegistry(),
        is_developer_mode_enabled=False,
    )


def _make_common(library: PtyChiReconstructorLibrary) -> PtyChiCommon:
    return PtyChiCommon(
        library.settings,
        library.object_settings,
        library.probe_settings,
        library.probe_position_settings,
        library.opr_settings,
    )


def _make_algorithms(library: PtyChiReconstructorLibrary) -> list[PtyChiAlgorithm]:
    """Instantiate the algorithm wrappers directly against ``library``'s settings.

    DM is first because the two ``[0]`` assertions below rely on it —
    ``build_algorithms`` guarantees that ordering.
    """
    algorithms = build_algorithms(
        _make_common(library),
        dm_settings=library.dm_settings,
        raar_settings=library.raar_settings,
        pie_settings=library.pie_settings,
        lsqml_settings=library.lsqml_settings,
        autodiff_settings=library.autodiff_settings,
        bh_settings=library.bh_settings,
    )
    return list(algorithms.values())


def _build_all_task_options(library: PtyChiReconstructorLibrary, parameters: ReconstructInput):
    # Building each task-options tree runs pty-chi's Pydantic validators over
    # every sub-option object.
    return [
        algorithm.build_task_options(parameters.product) for algorithm in _make_algorithms(library)
    ]


def _numeric_parameters(library: PtyChiReconstructorLibrary):
    settings_groups = [
        library.settings,
        library.object_settings,
        library.probe_settings,
        library.probe_position_settings,
        library.opr_settings,
        library.autodiff_settings,
        library.bh_settings,
        library.dm_settings,
        library.lsqml_settings,
        library.pie_settings,
        library.raar_settings,
    ]
    for group in settings_groups:
        for name, attr in vars(group).items():
            if isinstance(attr, (RealParameter, IntegerParameter)):
                yield f'{type(group).__name__}.{name}', attr


def test_default_settings_build_valid_options() -> None:
    """Every reconstructor's default options must satisfy pty-chi's validators."""
    library = _make_library()
    _build_all_task_options(library, _make_reconstruct_input())


def test_boundary_values_build_valid_options() -> None:
    """Setting each numeric parameter to its declared min/max must stay valid.

    This catches ptychodus bounds that are looser than pty-chi's enforced
    constraints (e.g. an inclusive minimum of 0 where pty-chi requires ``gt=0``).
    """
    library = _make_library()
    parameters = _make_reconstruct_input()

    for label, parameter in _numeric_parameters(library):
        original = parameter.get_value()
        for bound in (parameter.get_minimum(), parameter.get_maximum()):
            if bound is None:
                continue
            parameter.set_value(bound)
            try:
                _build_all_task_options(library, parameters)
            except Exception as exc:  # noqa: BLE001 - surface the offending parameter
                pytest.fail(f'{label} = {bound!r} produced invalid pty-chi options: {exc}')
        parameter.set_value(original)


def test_hard_limits_are_serialized_as_lists() -> None:
    """Enabled magnitude/phase hard limits must be plain lists (pty-chi rejects ndarrays)."""
    library = _make_library()
    obj = library.object_settings
    obj.constrain_hard_limits.set_value(True)
    obj.constrain_hard_limits_enable_abs.set_value(True)
    obj.constrain_hard_limits_enable_phase.set_value(True)

    options = _make_algorithms(library)[0].build_task_options(_make_reconstruct_input().product)
    hard_limits = options.object_options.hard_limits_magnitude_phase
    assert isinstance(hard_limits.abs_lim, list)
    assert isinstance(hard_limits.phase_lim, list)
    # Other serializable-array object fields must also be plain lists/tuples.
    assert isinstance(options.object_options.position_origin_coords, (list, tuple))
    assert list(options.object_options.position_origin_coords) == [0.0, 0.0]
    assert (
        options.object_options.determine_position_origin_coords_by
        is ObjectPosOriginCoordsMethods.SPECIFIED
    )
    assert options.object_options.slice_spacings_m is None or isinstance(
        options.object_options.slice_spacings_m, (list, tuple)
    )


def test_compact_mode_clustering_stride_is_at_least_one() -> None:
    """Disabled compact-mode clustering must still yield a stride >= 1 for pty-chi."""
    library = _make_library()
    # Default (disabled) value is 0; pty-chi's stride field is now ge=1.
    options = _make_algorithms(library)[0].build_task_options(_make_reconstruct_input().product)
    assert options.reconstructor_options.compact_mode_update_clustering_stride >= 1
    assert options.reconstructor_options.compact_mode_update_clustering is False


def test_built_options_carry_no_task_data() -> None:
    """Options must be settings-only; pty-chi deprecates carrying task data in them.

    pty-chi still honours the option fields, but warns per field and documents
    them as temporarily supported. Those warnings fire child-side and never
    reach the parent, so these ``is None`` assertions are the guard that keeps
    ptychodus off the deprecated path.
    """
    library = _make_library()
    parameters = _make_reconstruct_input()

    for algorithm in _make_algorithms(library):
        options = algorithm.build_task_options(parameters.product)
        assert options.data_options.data is None
        assert options.data_options.valid_pixel_mask is None
        assert options.object_options.initial_guess is None
        assert options.probe_options.initial_guess is None
        assert options.probe_position_options.position_x_px is None
        assert options.probe_position_options.position_y_px is None
        assert options.opr_mode_weight_options.initial_weights is None


def _make_probes(num_coherent_modes: int, opr_weights: numpy.ndarray | None) -> ProbeSequence:
    rng = numpy.random.default_rng(1)
    shape = (num_coherent_modes, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)
    return ProbeSequence(
        array=(rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
            numpy.complex128
        ),
        opr_weights=opr_weights,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
    )


def test_initial_opr_mode_weights_falls_back_to_primary_mode() -> None:
    """A probe with no OPR weights must yield all weight on the primary mode.

    ``reconstruct_with_ptychi`` runs only inside the GPU subprocess, so this
    fallback is otherwise reachable only from a real reconstruction. Importing
    the module parent-side is safe: it imports ``ptychi.api.options.task`` at
    module scope and defers ``PtychographyTask`` into the function body.
    """
    probes = _make_probes(3, opr_weights=None)

    with pytest.raises(ValueError):
        probes.get_opr_weights()

    weights = _initial_opr_mode_weights(probes)
    assert weights.shape == (probes.num_coherent_modes,)
    assert weights == pytest.approx([1.0, 0.0, 0.0])


def test_initial_opr_mode_weights_passes_through_existing_weights() -> None:
    """A probe that carries OPR weights must have them used as-is, not replaced."""
    opr_weights = numpy.random.default_rng(2).random((NUM_PATTERNS, 2))
    probes = _make_probes(2, opr_weights=opr_weights)

    assert _initial_opr_mode_weights(probes) is probes.get_opr_weights()


def test_far_field_propagation_flag_is_read_by_value() -> None:
    """The far-field checkbox must select the propagation mode.

    Regression test: the flag used to be read as a ``BooleanParameter`` object
    rather than via ``.get_value()``, so it was always truthy and the distance
    was pinned to infinity no matter what the user selected.

    ``data_options`` now carries only the mode -- infinity for far field, NaN as
    the near-field placeholder -- because the distance itself belongs to the
    product. See ``test_near_field_distance_comes_from_the_product``.
    """
    library = _make_library()
    common = _make_common(library)

    library.settings.use_far_field_propagation.set_value(True)
    assert common.data_options().free_space_propagation_distance_m == numpy.inf

    library.settings.use_far_field_propagation.set_value(False)
    assert math.isnan(common.data_options().free_space_propagation_distance_m)


def test_near_field_distance_comes_from_the_product() -> None:
    """The near-field placeholder must be replaced by the product's detector distance."""
    library = _make_library()
    parameters = _make_reconstruct_input()

    library.settings.use_far_field_propagation.set_value(False)
    options = _make_algorithms(library)[0].build_task_options(parameters.product)
    assert options.data_options.free_space_propagation_distance_m == pytest.approx(
        parameters.product.metadata.detector_distance_m
    )

    library.settings.use_far_field_propagation.set_value(True)
    options = _make_algorithms(library)[0].build_task_options(parameters.product)
    assert options.data_options.free_space_propagation_distance_m == numpy.inf


def test_options_round_trip_through_the_wire_format() -> None:
    """Serializing and reloading must preserve the algorithm-specific subclass.

    ``load_from_dict`` resolves nested options through declared field
    annotations, so loading into a plain ``PtychographyTaskOptions`` would
    downcast ``reconstructor_options`` to its base class and silently drop
    every algorithm-specific field. The subclass check below is what catches
    that.
    """
    library = _make_library()

    for task_options in _build_all_task_options(library, _make_reconstruct_input()):
        loaded = load_task_options(dump_task_options(task_options))

        assert type(loaded) is type(task_options)
        # Compare the parsed wire content rather than get_dict() directly:
        # pydantic coerces int defaults to float on load and JSON has no
        # tuples, so a raw dict comparison would test those artifacts.
        assert json.loads(dump_task_options(loaded)) == json.loads(dump_task_options(task_options))


def test_lsqml_algorithm_specific_field_survives_round_trip() -> None:
    """A field that exists only on the subclass must come back with its value."""
    library = _make_library()
    library.lsqml_settings.momentum_acceleration_gain.set_value(0.75)

    algorithm = LSQMLAlgorithm(_make_common(library), library.lsqml_settings)
    task_options = algorithm.build_task_options(_make_reconstruct_input().product)

    loaded = load_task_options(dump_task_options(task_options))

    assert loaded.reconstructor_options.momentum_acceleration_gain == pytest.approx(0.75)


def test_algorithm_task_options_class_matches_built_options() -> None:
    """Each algorithm must build the ``task_options_cls`` its spec names.

    Fails when a subclass sets ``spec.task_options_cls`` to a class whose
    ``get_reconstructor_type()`` disagrees with the expected reconstructor, and
    when two subclasses map to the same reconstructor type.
    """
    library = _make_library()
    parameters = _make_reconstruct_input()
    common = _make_common(library)

    cases: list[tuple[PtyChiAlgorithm, Reconstructors]] = [
        (DMAlgorithm(common, library.dm_settings), Reconstructors.DM),
        (RAARAlgorithm(common, library.raar_settings), Reconstructors.RAAR),
        (PIEAlgorithm(common, library.pie_settings), Reconstructors.PIE),
        (EPIEAlgorithm(common, library.pie_settings), Reconstructors.EPIE),
        (RPIEAlgorithm(common, library.pie_settings), Reconstructors.RPIE),
        (LSQMLAlgorithm(common, library.lsqml_settings), Reconstructors.LSQML),
        (AutodiffAlgorithm(common, library.autodiff_settings), Reconstructors.AD_PTYCHO),
        (BHAlgorithm(common, library.bh_settings), Reconstructors.BH),
    ]

    reconstructors_seen: set[Reconstructors] = set()

    for algorithm, expected_reconstructor in cases:
        options = algorithm.build_task_options(parameters.product)

        assert type(options) is type(algorithm).spec.task_options_cls
        assert options.reconstructor_options.get_reconstructor_type() == expected_reconstructor
        assert expected_reconstructor not in reconstructors_seen
        reconstructors_seen.add(expected_reconstructor)


def test_wire_format_envelope_carries_reconstructor() -> None:
    """The dumped JSON must be an envelope carrying the reconstructor token.

    Locks in the new wire format so a future refactor cannot silently revert
    to a bare-dict payload (which would let PIE/ePIE/rPIE alias each other).
    """
    library = _make_library()
    algorithm = LSQMLAlgorithm(_make_common(library), library.lsqml_settings)
    task_options = algorithm.build_task_options(_make_reconstruct_input().product)

    envelope = json.loads(dump_task_options(task_options))

    assert set(envelope) == {'reconstructor', 'options'}
    assert envelope['reconstructor'] == Reconstructors.LSQML.value
    assert isinstance(envelope['options'], dict)


def test_load_task_options_rejects_a_non_object_envelope() -> None:
    with pytest.raises(ValueError):
        load_task_options('[1, 2, 3]')


def test_load_task_options_rejects_an_unknown_reconstructor() -> None:
    with pytest.raises(ValueError):
        load_task_options(json.dumps({'reconstructor': 'not-a-real-algorithm', 'options': {}}))


def test_load_task_options_rejects_a_missing_reconstructor_field() -> None:
    with pytest.raises(ValueError):
        load_task_options(json.dumps({'options': {}}))


def test_load_task_options_rejects_a_non_object_options_field() -> None:
    with pytest.raises(ValueError):
        load_task_options(
            json.dumps({'reconstructor': Reconstructors.LSQML.value, 'options': [1, 2, 3]})
        )


# --- align_task_options_with_product ---------------------------------------
#
# Each entry names one field the helper owns, so dropping any single write
# fails a named parametrized case rather than a generic assertion. Nothing
# outside this table may set these fields; see
# ``test_common_kwargs_carry_no_product_derived_fields``.
_PRODUCT_DERIVED_FIELDS = [
    ('object_options.pixel_size_m', PIXEL_M),
    ('object_options.pixel_size_aspect_ratio', 1.0),
    ('object_options.slice_spacings_m', None),
    (
        'object_options.determine_position_origin_coords_by',
        ObjectPosOriginCoordsMethods.SPECIFIED,
    ),
    ('object_options.position_origin_coords', [0.0, 0.0]),
    ('data_options.wavelength_m', _make_reconstruct_input().product.metadata.probe_wavelength_m),
    ('data_options.free_space_propagation_distance_m', numpy.inf),
    ('probe_options.power_constraint.probe_power', PROBE_PHOTON_COUNT),
]


def _assert_field_equals(options, path: str, expected) -> None:
    actual = operator.attrgetter(path)(options)

    if isinstance(expected, (list, tuple)):
        assert list(actual) == list(expected), path
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected), path
    else:
        assert actual == expected, path


@pytest.fixture(scope='module')
def built_options_by_algorithm() -> list[tuple[str, object]]:
    """Options for all eight algorithms, built once.

    Module-scoped because ``_make_library`` spawns a device-probe subprocess
    that dominates this file's runtime; the parametrized test below reads these
    options without mutating settings, so one build serves every case.
    """
    library = _make_library()
    product = _make_reconstruct_input().product
    return [
        (type(algorithm).__name__, algorithm.build_task_options(product))
        for algorithm in _make_algorithms(library)
    ]


@pytest.mark.parametrize(('path', 'expected'), _PRODUCT_DERIVED_FIELDS, ids=lambda v: str(v)[:48])
def test_every_product_derived_field_is_applied(
    path: str, expected, built_options_by_algorithm
) -> None:
    """Built options must agree with the product on every field the helper owns.

    Checked across all eight algorithms: only that proves each algorithm's own
    ``*ObjectOptions`` / ``*ProbeOptions`` subclass accepts the assignment,
    which the type system does not check.
    """
    for algorithm_name, options in built_options_by_algorithm:
        try:
            _assert_field_equals(options, path, expected)
        except AssertionError as exc:
            pytest.fail(f'{algorithm_name}: {exc}')


def test_align_repairs_hand_built_options() -> None:
    """Bare options built by hand must come back fully described by the product.

    This is the shape ``scripts/lamni_reconstruct.py`` constructs. Before the
    helper existed it ran with pty-chi's defaults: a SUPPORT position origin
    that displaced every probe position by half the object canvas, plus a
    1e-9 m wavelength and a 1.0 m object pixel.
    """
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()

    aligned = align_task_options_with_product(options, parameters.product)

    for path, expected in _PRODUCT_DERIVED_FIELDS:
        _assert_field_equals(aligned, path, expected)

    aligned.check()


def test_align_does_not_mutate_its_argument() -> None:
    """The caller keeps the original so it can diff before against after."""
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    before = dump_task_options(options)

    aligned = align_task_options_with_product(options, parameters.product)

    assert aligned is not options
    assert aligned.object_options is not options.object_options
    assert dump_task_options(options) == before
    assert dump_task_options(aligned) != before


def test_align_is_idempotent(caplog) -> None:
    """Re-aligning against the same product changes nothing and warns about nothing.

    ``scripts/ptychodus_reconstruct.py`` re-aligns child-side over options its
    launcher already aligned; that second pass must be silent.
    """
    parameters = _make_reconstruct_input()
    once = align_task_options_with_product(LSQMLOptions(), parameters.product)

    with caplog.at_level(logging.WARNING, logger='ptychodus.model.ptychi.task'):
        twice = align_task_options_with_product(once, parameters.product)

    assert dump_task_options(twice) == dump_task_options(once)
    assert caplog.records == []


def test_align_warns_only_when_it_overrides_a_caller_value(caplog) -> None:
    """A non-default incoming value that disagrees is a warning; a default is not."""
    parameters = _make_reconstruct_input()

    options = LSQMLOptions()
    options.object_options.pixel_size_m = 5.0e-9

    with caplog.at_level(logging.WARNING, logger='ptychodus.model.ptychi.task'):
        aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.object_options.pixel_size_m == pytest.approx(PIXEL_M)
    warnings = [record for record in caplog.records if 'pixel_size_m' in record.getMessage()]
    assert len(warnings) == 1

    # The same field left at pty-chi's default is overwritten silently.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='ptychodus.model.ptychi.task'):
        align_task_options_with_product(LSQMLOptions(), parameters.product)

    assert caplog.records == []


def test_align_preserves_a_far_field_propagation_distance() -> None:
    """An infinite incoming distance is the caller's far-field choice, not a placeholder."""
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    assert options.data_options.free_space_propagation_distance_m == numpy.inf

    aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.data_options.free_space_propagation_distance_m == numpy.inf


def test_align_replaces_the_near_field_placeholder() -> None:
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    options.data_options.free_space_propagation_distance_m = math.nan

    aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.data_options.free_space_propagation_distance_m == pytest.approx(
        DETECTOR_DISTANCE_M
    )


def test_align_reads_a_non_square_object_pixel_aspect_ratio() -> None:
    """The shared fixture is square, so aspect ratio needs its own witness.

    A square object gives 1.0, which is also pty-chi's default -- an assertion
    on it would pass even if the helper stopped writing the field.
    """
    parameters = _make_reconstruct_input()
    obj = parameters.product.object_
    wide = Object(
        array=obj.get_array(),
        pixel_geometry=PixelGeometry(width_m=2.0 * PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[],
    )
    product = dataclasses.replace(parameters.product, object_=wide)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.object_options.pixel_size_m == pytest.approx(2.0 * PIXEL_M)
    assert aligned.object_options.pixel_size_aspect_ratio == pytest.approx(2.0)


def test_slice_spacings_from_an_ndarray_product_are_lists() -> None:
    """A multislice product read back from HDF5 carries an ndarray, not a list."""
    parameters = _make_reconstruct_input()
    rng = numpy.random.default_rng(1)
    multislice = Object(
        array=(
            rng.standard_normal((2, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
            + 1j * rng.standard_normal((2, OBJ_HEIGHT_PX, OBJ_WIDTH_PX))
        ).astype(numpy.complex128),
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=numpy.array([3.0e-8]),
    )
    product = dataclasses.replace(parameters.product, object_=multislice)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert isinstance(aligned.object_options.slice_spacings_m, list)
    assert aligned.object_options.slice_spacings_m == pytest.approx([3.0e-8])


def test_common_kwargs_carry_no_product_derived_fields() -> None:
    """PtyChiCommon must not re-acquire an owner for anything the helper writes.

    A second writer would make the helper's override warning fire on every
    reconstruction.
    """
    library = _make_library()
    common = _make_common(library)

    object_keys = set(common.object_kwargs())
    probe_keys = set(common.probe_kwargs())
    data_options = common.data_options()

    assert not object_keys & {
        'pixel_size_m',
        'pixel_size_aspect_ratio',
        'slice_spacings_m',
        'determine_position_origin_coords_by',
        'position_origin_coords',
    }
    assert 'probe_power' not in probe_keys
    assert data_options.wavelength_m == PtychographyDataOptions().wavelength_m
    assert not math.isfinite(data_options.free_space_propagation_distance_m)
