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

from collections.abc import Iterator
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
from ptychi.api import (
    RECONSTRUCTOR_OPTIONS_MAP,
    LSQMLOptions,
    LSQMLReconstructorOptions,
    ObjectPosOriginCoordsMethods,
    Reconstructors,
)
from ptychi.api.options.data import PtychographyDataOptions
import torch

from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary
from ptychodus.model.ptychi.task import _recover_layer_spacing_m

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


@pytest.fixture(scope='module', autouse=True)
def _stub_the_device_probe() -> Iterator[None]:
    """Skip the spawned GPU enumeration; nothing here reads the device list.

    PtyChiDeviceRepository probes devices in a child process that imports
    torch -- about 17 seconds, once per library construction, and this module
    builds sixteen. The repository resolves the probe on its module at call
    time, so patching the module attribute takes effect.

    Module-scoped, and therefore not the function-scoped ``monkeypatch``
    fixture: ``built_options_by_algorithm`` below is itself module-scoped and
    would build its library before any function-scoped patch was in place.
    """
    from ptychodus.model.ptychi import device

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(device, '_probe_devices_via_subprocess', lambda: ['cpu'])
        yield


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


def test_no_setting_selects_the_propagation_mode() -> None:
    """The regime is a property of the experiment, so the reconstructor settings must
    not carry a second switch for it.

    ``data_options`` leaves ``free_space_propagation_distance_m`` at pty-chi's own
    default; the product supplies the value during alignment. See
    ``test_near_field_distance_comes_from_the_product``.
    """
    library = _make_library()
    common = _make_common(library)

    assert not hasattr(library.settings, 'use_far_field_propagation')
    assert (
        common.data_options().free_space_propagation_distance_m
        == PtychographyDataOptions().free_space_propagation_distance_m
    )


def test_near_field_distance_comes_from_the_product() -> None:
    """A product declaring near field supplies the distance; far field stays infinite."""
    library = _make_library()
    parameters = _make_reconstruct_input()

    near_field = _with_far_field(parameters.product, False)
    options = _make_algorithms(library)[0].build_task_options(near_field)
    assert options.data_options.free_space_propagation_distance_m == pytest.approx(
        near_field.metadata.detector_distance_m
    )

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

    # Narrowed, not cast: the round trip must preserve the LSQML subclass, and only
    # that subclass carries the field.
    assert isinstance(loaded.reconstructor_options, LSQMLReconstructorOptions)
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


def test_spec_option_classes_match_the_upstream_reconstructor_map() -> None:
    """Each ``_Spec`` must name the option classes pty-chi maps to its reconstructor.

    ``algorithms.py`` hand-writes 48 option-class references and mypy cannot
    check them -- ``ptychi.*`` is in ``ignore_missing_imports``. pty-chi 2.1.0
    exports ``RECONSTRUCTOR_OPTIONS_MAP``, so the hand-written table can now be
    checked against upstream instead of drifting silently.

    The map covers only the five nested option classes; ``task_options_cls`` and
    the display name stay ptychodus's own, which is why this guards the spec
    rather than replacing it.
    """
    library = _make_library()

    for algorithm in _make_algorithms(library):
        spec = type(algorithm).spec
        reconstructor = spec.reconstructor_options_cls().get_reconstructor_type()
        expected = RECONSTRUCTOR_OPTIONS_MAP[reconstructor]

        assert spec.object_options_cls is expected['object_options'], spec.display_name
        assert spec.probe_options_cls is expected['probe_options'], spec.display_name
        assert spec.probe_position_options_cls is expected['probe_position_options'], (
            spec.display_name
        )
        assert spec.opr_options_cls is expected['opr_mode_weight_options'], spec.display_name
        assert spec.reconstructor_options_cls is expected['reconstructor_options'], (
            spec.display_name
        )


def test_wire_format_carries_the_options_class_name() -> None:
    """The dumped JSON must be a bare options dict stamped with its class name.

    Locks in the wire format so a future refactor cannot silently drop the stamp
    (which would let PIE/ePIE/rPIE alias each other, since their dicts are
    otherwise identical).
    """
    library = _make_library()
    algorithm = LSQMLAlgorithm(_make_common(library), library.lsqml_settings)
    task_options = algorithm.build_task_options(_make_reconstruct_input().product)

    options_dict = json.loads(dump_task_options(task_options))

    assert options_dict['options_class_name'] == 'LSQMLOptions'
    assert 'reconstructor_options' in options_dict
    # The old ptychodus envelope is gone; pty-chi's own stamp replaced it.
    assert 'reconstructor' not in options_dict
    assert 'options' not in options_dict


def test_wire_format_distinguishes_pie_variants() -> None:
    """PIE, ePIE and rPIE must each carry a distinct stamp.

    Their serialized field sets are identical, so the stamp is the only thing
    keeping a dumped ePIE from reloading as PIE and running a different
    algorithm. This is what the hand-rolled envelope used to guarantee.
    """
    library = _make_library()
    common = _make_common(library)
    product = _make_reconstruct_input().product

    algorithms = [
        PIEAlgorithm(common, library.pie_settings),
        EPIEAlgorithm(common, library.pie_settings),
        RPIEAlgorithm(common, library.pie_settings),
    ]
    stamps = set()

    for algorithm in algorithms:
        task_options = algorithm.build_task_options(product)
        loaded = load_task_options(dump_task_options(task_options))

        assert type(loaded) is type(task_options)
        stamps.add(json.loads(dump_task_options(task_options))['options_class_name'])

    assert stamps == {'PIEOptions', 'EPIEOptions', 'RPIEOptions'}


def test_load_task_options_rejects_a_non_object_payload() -> None:
    with pytest.raises(ValueError):
        load_task_options('[1, 2, 3]')


@pytest.mark.parametrize(
    'options_class_name',
    [
        'NotARealOptionsClass',
        # Real ``ptychi.api`` attributes that are not task options. Resolution
        # goes through the module, so the subclass test is the only thing
        # stopping any of these from being instantiated and loaded into.
        'Reconstructors',
        'ObjectOptions',
        'PtychographyTaskOptions',
    ],
)
def test_load_task_options_rejects_a_name_that_is_not_a_task_options_class(
    options_class_name: str,
) -> None:
    with pytest.raises(ValueError):
        load_task_options(json.dumps({'options_class_name': options_class_name}))


def test_load_task_options_rejects_a_missing_options_class_name() -> None:
    with pytest.raises(ValueError):
        load_task_options(json.dumps({'reconstructor_options': {}}))


def test_load_task_options_rejects_a_mismatched_nested_options_class() -> None:
    """A payload assembled from mismatched parts must not load quietly.

    Only ``strict=True`` catches this: the top-level stamp still says
    ``LSQMLOptions``, so class selection succeeds and it is the nested check
    that has to fire.
    """
    options_dict = json.loads(dump_task_options(LSQMLOptions()))
    options_dict['reconstructor_options']['options_class_name'] = 'DMReconstructorOptions'

    with pytest.raises(ValueError):
        load_task_options(json.dumps(options_dict))


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
    # None means "inherit the object value": the test product's probe and object
    # sampling agree, and writing None is what keeps pty-chi's no-resampling
    # fast path. See test_align_reads_a_differing_probe_pixel_geometry.
    ('probe_options.pixel_size_m', None),
    ('probe_options.pixel_size_aspect_ratio', None),
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

    This is the shape a hand-assembled options object has. Before the helper
    existed such an object ran with pty-chi's defaults: a SUPPORT position origin
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
    """Re-aligning against the same product changes nothing and overrides nothing.

    A child process re-aligns over options its launcher already aligned; that second
    pass must emit no override warning. The regime self-consistency warning is a
    property of the product rather than of the options diff, so it would fire on both
    passes -- this fixture is far-field consistent, so neither pass emits it.
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
    """A far-field product keeps pty-chi's infinite distance."""
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    assert options.data_options.free_space_propagation_distance_m == numpy.inf

    aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.data_options.free_space_propagation_distance_m == numpy.inf


def test_align_writes_the_near_field_distance_from_the_product() -> None:
    product = _with_far_field(_make_reconstruct_input().product, False)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.data_options.free_space_propagation_distance_m == pytest.approx(
        DETECTOR_DISTANCE_M
    )


def test_align_overwrites_a_contradicting_caller_distance_and_warns(caplog) -> None:
    """A ptychi_options.json that hard-codes a distance disagreeing with the product
    is corrected, loudly -- the product owns the field outright."""
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    options.data_options.free_space_propagation_distance_m = 0.5

    with caplog.at_level(logging.WARNING, logger='ptychodus.model.ptychi.task'):
        aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.data_options.free_space_propagation_distance_m == numpy.inf
    warnings = [
        record
        for record in caplog.records
        if 'free_space_propagation_distance_m' in record.getMessage()
    ]
    assert len(warnings) == 1


def _with_far_field(product: Product, far_field: bool) -> Product:
    return dataclasses.replace(
        product, metadata=dataclasses.replace(product.metadata, far_field=far_field)
    )


def _make_cone_beam_product(focus_object_distance_m: float) -> Product:
    """The shared fixture declares no focusing optic; these need one.

    Near field as well: these pin the demagnified propagation distance, which is only
    written for a product that declares near-field propagation.
    """
    product = _make_reconstruct_input().product
    return dataclasses.replace(
        product,
        metadata=dataclasses.replace(
            product.metadata,
            focus_object_distance_m=focus_object_distance_m,
            far_field=False,
        ),
    )


def test_align_demagnifies_the_near_field_distance_for_a_cone_beam() -> None:
    """pty-chi propagates in the equivalent parallel-beam geometry, so a magnifying
    geometry contributes z_d / M rather than the raw detector distance.
    """
    focus_m = 5e-3
    product = _make_cone_beam_product(focus_m)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    magnification = (DETECTOR_DISTANCE_M - focus_m) / focus_m
    assert aligned.data_options.free_space_propagation_distance_m == pytest.approx(
        DETECTOR_DISTANCE_M / magnification
    )


def test_align_distinguishes_converging_from_diverging_illumination() -> None:
    """The sign of the focus coordinate is a real geometry choice, so it must reach
    the propagation distance rather than being absorbed as a magnitude.
    """
    options = LSQMLOptions()

    converging = align_task_options_with_product(options, _make_cone_beam_product(5e-3))
    diverging = align_task_options_with_product(options, _make_cone_beam_product(-5e-3))

    assert converging.data_options.free_space_propagation_distance_m != pytest.approx(
        diverging.data_options.free_space_propagation_distance_m
    )


def test_align_without_a_focusing_optic_uses_the_raw_detector_distance() -> None:
    """The parallel-beam path must be untouched by the cone-beam support."""
    aligned = align_task_options_with_product(LSQMLOptions(), _make_cone_beam_product(0.0))

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


def test_align_reads_a_differing_probe_pixel_geometry() -> None:
    """A probe sampled differently from the object must say so explicitly.

    pty-chi resamples the object onto the probe grid when the two disagree, and
    it only knows to do that if the probe fields are written. The shared fixture
    has matching geometries, so this needs its own witness.
    """
    parameters = _make_reconstruct_input()
    probes = parameters.product.probes
    resampled = ProbeSequence(
        array=probes.get_array(),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=2.0 * PIXEL_M, height_m=4.0 * PIXEL_M),
    )
    product = dataclasses.replace(parameters.product, probes=resampled)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.probe_options.pixel_size_m == pytest.approx(2.0 * PIXEL_M)
    assert aligned.probe_options.pixel_size_aspect_ratio == pytest.approx(0.5)
    # The object keeps its own sampling; only the probe fields move.
    assert aligned.object_options.pixel_size_m == pytest.approx(PIXEL_M)
    assert aligned.object_options.pixel_size_aspect_ratio == pytest.approx(1.0)


def test_align_inherits_probe_pixel_width_when_only_the_aspect_ratio_differs() -> None:
    """pty-chi inherits each omitted probe field independently, so we write them so."""
    parameters = _make_reconstruct_input()
    probes = parameters.product.probes
    tall = ProbeSequence(
        array=probes.get_array(),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=2.0 * PIXEL_M),
    )
    product = dataclasses.replace(parameters.product, probes=tall)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.probe_options.pixel_size_m is None
    assert aligned.probe_options.pixel_size_aspect_ratio == pytest.approx(0.5)


def test_align_tolerates_a_probe_without_pixel_geometry() -> None:
    """A probe that cannot report its sampling must not make alignment raise.

    ``ProbeSequence.get_pixel_geometry`` raises when unset, and such a product
    reconstructed fine before the probe fields existed; it must still align.
    """
    parameters = _make_reconstruct_input()
    probes = parameters.product.probes
    ungeometried = ProbeSequence(
        array=probes.get_array(),
        opr_weights=None,
        pixel_geometry=None,
    )
    product = dataclasses.replace(parameters.product, probes=ungeometried)

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.probe_options.pixel_size_m is None
    assert aligned.probe_options.pixel_size_aspect_ratio is None


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
        # An ndarray on purpose: converting it to a list is what is under test.
        layer_spacing_m=numpy.array([3.0e-8]),  # type: ignore[arg-type]
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
    assert not probe_keys & {'probe_power', 'pixel_size_m', 'pixel_size_aspect_ratio'}
    assert data_options.wavelength_m == PtychographyDataOptions().wavelength_m
    assert not math.isfinite(data_options.free_space_propagation_distance_m)


# ---------------------------------------------------------------------------
# Slice-spacing recovery
# ---------------------------------------------------------------------------


class _StubSliceSpacings:
    def __init__(self, values_m: list[float]) -> None:
        self.data = torch.tensor(values_m)


class _StubTaskObject:
    def __init__(self, values_m: list[float]) -> None:
        self.slice_spacings = _StubSliceSpacings(values_m)


class _StubTask:
    def __init__(self, task_object: object) -> None:
        self.object = task_object


def _make_layered_object(num_layers: int, spacing_m: float) -> Object:
    return Object(
        array=numpy.ones((num_layers, 4, 4), dtype=complex),
        pixel_geometry=PixelGeometry(width_m=1.0e-9, height_m=1.0e-9),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[spacing_m] * (num_layers - 1),
    )


def test_recover_layer_spacing_prefers_the_optimized_values() -> None:
    """pty-chi can optimize slice spacings, so the output must carry what it ended with."""
    object_in = _make_layered_object(3, 1.0e-6)
    task = _StubTask(_StubTaskObject([2.0e-6, 3.0e-6]))

    recovered = _recover_layer_spacing_m(task, object_in)

    numpy.testing.assert_allclose(recovered, [2.0e-6, 3.0e-6], atol=1.0e-15)


def test_recover_layer_spacing_falls_back_when_the_attribute_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The access path is pty-chi private structure, so a version that moves it must
    degrade to the input spacing rather than crash the reconstruction."""
    object_in = _make_layered_object(3, 1.0e-6)

    with caplog.at_level(logging.WARNING):
        recovered = _recover_layer_spacing_m(_StubTask(object()), object_in)

    numpy.testing.assert_allclose(recovered, [1.0e-6, 1.0e-6], atol=1.0e-15)
    assert 'Could not read slice spacings back' in caplog.text


def test_recover_layer_spacing_falls_back_on_a_length_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A wrong-length spacing would make the Object constructor raise, turning a cosmetic
    loss into a crash."""
    object_in = _make_layered_object(3, 1.0e-6)
    task = _StubTask(_StubTaskObject([2.0e-6, 3.0e-6, 4.0e-6]))

    with caplog.at_level(logging.WARNING):
        recovered = _recover_layer_spacing_m(task, object_in)

    numpy.testing.assert_allclose(recovered, [1.0e-6, 1.0e-6], atol=1.0e-15)
    assert 'slice spacing(s) for a 3-layer object' in caplog.text


def test_recover_layer_spacing_single_layer_needs_no_readback() -> None:
    """A single-layer object has no gaps, and pty-chi stores a placeholder there."""
    object_in = _make_layered_object(1, 0.0)

    assert _recover_layer_spacing_m(_StubTask(_StubTaskObject([0.0])), object_in) == []
