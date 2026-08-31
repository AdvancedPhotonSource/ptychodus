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
from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary

# Concrete (child-side) reconstructor classes plus the options helper. This test
# is an in-process validator that must reach into pty-chi's Pydantic constructors
# directly, so it deliberately does the imports the parent avoids. This is safe
# because the file already does ``pytest.importorskip('ptychi')`` above.
from ptychodus.model.ptychi._subprocess import _initial_opr_mode_weights  # noqa: E402
from ptychodus.model.ptychi.autodiff import AutodiffReconstructor  # noqa: E402
from ptychodus.model.ptychi.bh import BHReconstructor  # noqa: E402
from ptychodus.model.ptychi.dm import DMReconstructor  # noqa: E402
from ptychodus.model.ptychi.epie import EPIEReconstructor  # noqa: E402
from ptychodus.model.ptychi.helper import PtyChiOptionsHelper  # noqa: E402
from ptychodus.model.ptychi.lsqml import LSQMLReconstructor  # noqa: E402
from ptychodus.model.ptychi.pie import PIEReconstructor  # noqa: E402
from ptychodus.model.ptychi.raar import RAARReconstructor  # noqa: E402
from ptychodus.model.ptychi.rpie import RPIEReconstructor  # noqa: E402

PIXEL_M = 1.0e-9
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
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
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
        [
            ProbePosition(index=i, coordinate_x_m=i * PIXEL_M, coordinate_y_m=-i * PIXEL_M)
            for i in range(NUM_PATTERNS)
        ]
    )
    product = Product(
        metadata=ProductMetadata(
            name='test',
            comments='',
            detector_distance_m=1.0,
            probe_energy_eV=10_000.0,
            probe_photon_count=1.0,
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


def _make_options_helper(library: PtyChiReconstructorLibrary) -> PtyChiOptionsHelper:
    return PtyChiOptionsHelper(
        library.settings,
        library.object_settings,
        library.probe_settings,
        library.probe_position_settings,
        library.opr_settings,
    )


def _make_concrete_reconstructors(library: PtyChiReconstructorLibrary) -> list:
    """Instantiate the child-side algorithm classes directly against ``library``'s settings.

    The parent-side ``library.reconstructor_list`` now holds
    :class:`SubprocessReconstructor` shells that hide ``_create_task_options``
    behind a process boundary. To validate defaults / bounds we need the
    concrete classes, so build them here using the same wiring the child does.

    Index 0 must stay ``DMReconstructor``: two tests below index this list
    positionally to get "the DM one".
    """
    helper = _make_options_helper(library)
    return [
        DMReconstructor(helper, library.dm_settings),
        RAARReconstructor(helper, library.raar_settings),
        PIEReconstructor(helper, library.pie_settings),
        EPIEReconstructor(helper, library.pie_settings),
        RPIEReconstructor(helper, library.pie_settings),
        LSQMLReconstructor(helper, library.lsqml_settings),
        AutodiffReconstructor(helper, library.autodiff_settings),
        BHReconstructor(helper, library.bh_settings),
    ]


def _build_all_task_options(library: PtyChiReconstructorLibrary, parameters: ReconstructInput):
    for reconstructor in _make_concrete_reconstructors(library):
        # Every ptychi reconstructor exposes ``_create_task_options``; building it
        # runs pty-chi's Pydantic validators over all sub-option objects.
        reconstructor._create_task_options(parameters)  # type: ignore[attr-defined]


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

    options = _make_concrete_reconstructors(library)[0]._create_task_options(  # type: ignore[attr-defined]
        _make_reconstruct_input()
    )
    hard_limits = options.object_options.hard_limits_magnitude_phase
    assert isinstance(hard_limits.abs_lim, list)
    assert isinstance(hard_limits.phase_lim, list)
    # Other serializable-array object fields must also be plain lists/tuples.
    assert isinstance(options.object_options.position_origin_coords, (list, tuple))
    assert options.object_options.slice_spacings_m is None or isinstance(
        options.object_options.slice_spacings_m, (list, tuple)
    )


def test_compact_mode_clustering_stride_is_at_least_one() -> None:
    """Disabled compact-mode clustering must still yield a stride >= 1 for pty-chi."""
    library = _make_library()
    # Default (disabled) value is 0; pty-chi's stride field is now ge=1.
    options = _make_concrete_reconstructors(library)[0]._create_task_options(  # type: ignore[attr-defined]
        _make_reconstruct_input()
    )
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

    for reconstructor in _make_concrete_reconstructors(library):
        options = reconstructor._create_task_options(parameters)  # type: ignore[attr-defined]
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
    """The far-field checkbox must select the propagation distance.

    Regression test: the flag used to be read as a ``BooleanParameter`` object
    rather than via ``.get_value()``, so it was always truthy and the distance
    was pinned to infinity no matter what the user selected.
    """
    library = _make_library()
    metadata = _make_reconstruct_input().product.metadata
    helper = _make_options_helper(library)

    library.settings.use_far_field_propagation.set_value(True)
    assert helper.create_data_options(metadata).free_space_propagation_distance_m == numpy.inf

    library.settings.use_far_field_propagation.set_value(False)
    assert helper.create_data_options(metadata).free_space_propagation_distance_m == pytest.approx(
        metadata.detector_distance_m
    )
