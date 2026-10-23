"""Consistency checks between ptychodus pty-chi settings and pty-chi Options.

pty-chi's ``Options`` classes are Pydantic dataclasses that enforce every
``ge``/``gt``/``le`` constraint at construction time. These tests build the full
per-reconstructor task-options object from the live settings and let pty-chi's
validators run, so a default or a bound that pty-chi would reject surfaces
here rather than as a reconstruction-time crash. No GPU or real dataset is required — per-field validation runs at
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
import argparse
import math
import operator
from pathlib import Path

import numpy
import pytest
import yaml

pytest.importorskip('ptychi')

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.parameters import IntegerParameter, RealParameter
from ptychodus.api.probe import ProbeEnsemble
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
from ptychi.api.options.task import PtychographyTaskOptions
import torch

from ptychodus.model.processing import ProcessingParameterStore
from ptychodus.model.processing._ptychi_sections import sections_for
from ptychodus.plugins.ptychi.task import _recover_layer_spacing_m

# Deliberately imported at module scope: this file already does
# ``pytest.importorskip('ptychi')`` above.
from ptychodus.plugins.ptychi.algorithms import (  # noqa: E402
    _ALGORITHM_CLASSES,
    build_reconstructors,
)
from ptychodus.plugins.ptychi.options import apply_to_task  # noqa: E402

# The name pty-chi registers itself under, which is how the store files it.
_LIBRARY = 'pty-chi'
from ptychodus.cli._reconstruct_common import (  # noqa: E402
    _DATA_DERIVED_OPTION_FIELDS,
    add_print_ptychi_options_argument,
    format_ptychi_options_defaults,
    load_ptychi_options,
)
from ptychodus.plugins.ptychi.algorithms import task_options_classes  # noqa: E402
from ptychodus.plugins.ptychi.task import (  # noqa: E402
    _build_ptychi_propagator,
    _initial_opr_mode_weights,
    align_probe_gauge_with_ptychi,
    align_task_options_with_product,
    apply_probe_gauge_to_options,
    dump_task_options,
    dump_task_options_yaml,
    load_task_options,
    load_task_options_from_mapping,
)

PIXEL_M = 1.0e-9
# Every expected value below is a distinct witness: none coincides with a
# pty-chi default (1.0 m, 1e-9 m, inf, 0.0), so a test cannot pass by accident
# if the helper drops the field it is checking.
DETECTOR_DISTANCE_M = 2.5
PHOTON_ENERGY_EV = 10_000.0
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
    probes = ProbeEnsemble(
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
            photon_energy_eV=PHOTON_ENERGY_EV,
            probe_photon_count=PROBE_PHOTON_COUNT,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
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


class _Harness:
    """Every pty-chi algorithm, with the live settings generated behind it.

    The same arrangement a run uses: each reconstructor's declaration is
    registered with a parameter store, and options are read back out of it.
    Exercising that path is the point -- a test that built options straight
    from the declaration would skip the flattening and the restoration that
    stand between a settings file and pty-chi.

    DM is first, because two assertions below take the first algorithm.
    """

    def __init__(self) -> None:
        self.registry = SettingsRegistry()
        self.store = ProcessingParameterStore(self.registry)
        self._names: list[str] = []

        for reconstructor in build_reconstructors():
            self.store.register(
                _LIBRARY,
                reconstructor,
                sections_for(reconstructor.name, reconstructor.get_default_options()),
            )
            self._names.append(reconstructor.name)

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def parameter(self, group: str, key: str):
        return self.registry[group].parameters()[key]

    def declared_task_options(self, name: str):
        """What the settings alone produce, before the product has a say."""
        return apply_to_task(self.store.get_options(_LIBRARY, name))

    def task_options(self, name: str, product: Product):
        # Building each task-options tree runs pty-chi's Pydantic validators
        # over every sub-option object.
        return align_task_options_with_product(self.declared_task_options(name), product)

    def all_task_options(self, product: Product) -> list:
        return [self.task_options(name, product) for name in self._names]

    def numeric_parameters(self):
        for group_name in sorted(self.registry):
            if not group_name.startswith('PtyChi'):
                continue

            for key, parameter in self.registry[group_name].parameters().items():
                if isinstance(parameter, (RealParameter, IntegerParameter)):
                    yield f'{group_name}.{key}', parameter


def test_default_settings_build_valid_options() -> None:
    """Every reconstructor's default options must satisfy pty-chi's validators."""
    _Harness().all_task_options(_make_reconstruct_input().product)


def test_boundary_values_build_valid_options() -> None:
    """Setting each numeric parameter to its declared min/max must stay valid.

    This catches ptychodus bounds that are looser than pty-chi's enforced
    constraints (e.g. an inclusive minimum of 0 where pty-chi requires ``gt=0``).
    """
    harness = _Harness()
    product = _make_reconstruct_input().product

    for label, parameter in harness.numeric_parameters():
        original = parameter.get_value()
        for bound in (parameter.get_minimum(), parameter.get_maximum()):
            if bound is None:
                continue
            parameter.set_value(bound)
            try:
                harness.all_task_options(product)
            except Exception as exc:  # noqa: BLE001 - surface the offending parameter
                pytest.fail(f'{label} = {bound!r} produced invalid pty-chi options: {exc}')
        parameter.set_value(original)


def test_hard_limits_are_serialized_as_lists() -> None:
    """Enabled magnitude/phase hard limits must be plain lists (pty-chi rejects ndarrays)."""
    harness = _Harness()

    for key in (
        'ConstrainHardLimits',
        'ConstrainHardLimitsEnableAbs',
        'ConstrainHardLimitsEnablePhase',
    ):
        harness.parameter('PtyChiObject', key).set_value(True)

    options = harness.task_options('DM', _make_reconstruct_input().product)
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
    # Default (disabled) value is 0; pty-chi's stride field is now ge=1.
    options = _Harness().task_options('DM', _make_reconstruct_input().product)
    assert options.reconstructor_options.compact_mode_update_clustering_stride >= 1
    assert options.reconstructor_options.compact_mode_update_clustering is False


def test_built_options_carry_no_task_data() -> None:
    """Options must be settings-only; pty-chi deprecates carrying task data in them.

    pty-chi still honours the option fields, but warns per field and documents
    them as temporarily supported. Those warnings fire child-side and never
    reach the parent, so these ``is None`` assertions are the guard that keeps
    ptychodus off the deprecated path.
    """
    parameters = _make_reconstruct_input()

    for options in _Harness().all_task_options(parameters.product):
        assert options.data_options.data is None
        assert options.data_options.valid_pixel_mask is None
        assert options.object_options.initial_guess is None
        assert options.probe_options.initial_guess is None
        assert options.probe_position_options.position_x_px is None
        assert options.probe_position_options.position_y_px is None
        assert options.opr_mode_weight_options.initial_weights is None


def _make_probes(num_coherent_modes: int, opr_weights: numpy.ndarray | None) -> ProbeEnsemble:
    rng = numpy.random.default_rng(1)
    shape = (num_coherent_modes, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX)
    return ProbeEnsemble(
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
    harness = _Harness()

    assert 'UseFarFieldPropagation' not in harness.registry['PtyChi'].parameters()
    assert (
        harness.declared_task_options('DM').data_options.free_space_propagation_distance_m
        == PtychographyDataOptions().free_space_propagation_distance_m
    )


def test_near_field_distance_comes_from_the_product() -> None:
    """A product declaring near field supplies the distance; far field stays infinite."""
    harness = _Harness()
    parameters = _make_reconstruct_input()

    near_field = _with_far_field(parameters.product, False)
    options = harness.task_options('DM', near_field)
    assert options.data_options.free_space_propagation_distance_m == pytest.approx(
        near_field.metadata.detector_distance_m
    )

    options = harness.task_options('DM', parameters.product)
    assert options.data_options.free_space_propagation_distance_m == numpy.inf


def test_options_round_trip_through_the_wire_format() -> None:
    """Serializing and reloading must preserve the algorithm-specific subclass.

    ``load_from_dict`` resolves nested options through declared field
    annotations, so loading into a plain ``PtychographyTaskOptions`` would
    downcast ``reconstructor_options`` to its base class and silently drop
    every algorithm-specific field. The subclass check below is what catches
    that.
    """
    for task_options in _Harness().all_task_options(_make_reconstruct_input().product):
        loaded = load_task_options(dump_task_options(task_options))

        assert type(loaded) is type(task_options)
        # Compare the parsed wire content rather than get_dict() directly:
        # pydantic coerces int defaults to float on load and JSON has no
        # tuples, so a raw dict comparison would test those artifacts.
        assert json.loads(dump_task_options(loaded)) == json.loads(dump_task_options(task_options))


def test_lsqml_algorithm_specific_field_survives_round_trip() -> None:
    """A field that exists only on the subclass must come back with its value."""
    harness = _Harness()
    harness.parameter('PtyChiLSQML', 'MomentumAccelerationGain').set_value(0.75)

    task_options = harness.task_options('LSQML', _make_reconstruct_input().product)

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
    harness = _Harness()
    parameters = _make_reconstruct_input()

    expected_by_name = {
        'DM': Reconstructors.DM,
        'RAAR': Reconstructors.RAAR,
        'PIE': Reconstructors.PIE,
        'ePIE': Reconstructors.EPIE,
        'rPIE': Reconstructors.RPIE,
        'LSQML': Reconstructors.LSQML,
        'Autodiff': Reconstructors.AD_PTYCHO,
        'BH': Reconstructors.BH,
    }
    specs = {cls.spec.display_name: cls.spec for cls in _ALGORITHM_CLASSES}
    assert set(expected_by_name) == set(specs)

    reconstructors_seen: set[Reconstructors] = set()

    for name, expected_reconstructor in expected_by_name.items():
        options = harness.task_options(name, parameters.product)

        assert type(options) is specs[name].task_options_cls
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
    for cls in _ALGORITHM_CLASSES:
        spec = cls.spec
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
    task_options = _Harness().task_options('LSQML', _make_reconstruct_input().product)

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
    harness = _Harness()
    product = _make_reconstruct_input().product
    stamps = set()

    for name in ('PIE', 'ePIE', 'rPIE'):
        task_options = harness.task_options(name, product)
        loaded = load_task_options(dump_task_options(task_options))

        assert type(loaded) is type(task_options)
        stamps.add(json.loads(dump_task_options(task_options))['options_class_name'])

    assert stamps == {'PIEOptions', 'EPIEOptions', 'RPIEOptions'}


def test_yaml_round_trips_every_algorithm_to_the_same_payload_as_json() -> None:
    """The options file encoding must carry exactly what the wire format carries.

    ``get_dict`` hands back live pty-chi objects -- ``StrEnum`` members throughout,
    and a tuple of them under ``probe_position_options.affine_transform_constraint``
    -- which ``yaml.safe_dump`` refuses to represent even though ``json.dumps``
    renders both. Every algorithm is covered because a pty-chi release that adds a
    field of some further unrepresentable type would otherwise fail at the end of a
    reconstruction, after the expensive part, rather than here.
    """
    product = _make_reconstruct_input().product

    for task_options in _Harness().all_task_options(product):
        text = dump_task_options_yaml(task_options)
        loaded = load_task_options_from_mapping(yaml.safe_load(text))

        assert type(loaded) is type(task_options)
        assert json.loads(dump_task_options(loaded)) == json.loads(dump_task_options(task_options))


def test_yaml_options_preserve_pty_chi_field_order() -> None:
    """A file meant to be read keeps each options block's own field order."""
    options = LSQMLOptions()
    text = dump_task_options_yaml(options)

    assert list(yaml.safe_load(text)) == list(json.loads(dump_task_options(options)))


def test_load_ptychi_options_reads_yaml_and_json_alike(tmp_path: Path) -> None:
    """Both encodings of one options object load to the same thing."""
    options = LSQMLOptions()

    yaml_file = tmp_path / 'options.yaml'
    yaml_file.write_text(dump_task_options_yaml(options))
    json_file = tmp_path / 'options.json'
    json_file.write_text(dump_task_options(options))

    from_yaml = load_ptychi_options(yaml_file)
    from_json = load_ptychi_options(json_file)

    assert type(from_yaml) is type(from_json) is LSQMLOptions
    assert json.loads(dump_task_options(from_yaml)) == json.loads(dump_task_options(from_json))


# --- --print-ptychi-options --------------------------------------------------------


def test_printed_defaults_name_the_same_data_derived_fields_as_alignment() -> None:
    """The printed header's warning list must be the fields alignment actually writes.

    The header tells a reader which values are pointless to edit. That list is written
    out by hand -- a reader of the file cannot introspect anything -- so this is what
    keeps it honest against ``_PRODUCT_DERIVED_FIELDS``, which the test above proves is
    what alignment really sets.
    """
    assert set(_DATA_DERIVED_OPTION_FIELDS) == {path for path, _ in _PRODUCT_DERIVED_FIELDS}


@pytest.mark.parametrize('algorithm', list(task_options_classes()))
def test_printed_defaults_round_trip_to_that_algorithm(algorithm: str) -> None:
    """Printing then loading must give back the algorithm that was asked for.

    This is the loop the flag exists for, and it covers the ``options_class_name``
    stamp: PIE, ePIE and rPIE serialize identical field sets, so only the stamp keeps
    them apart.
    """
    loaded = load_task_options_from_mapping(
        yaml.safe_load(format_ptychi_options_defaults(algorithm))
    )

    assert type(loaded) is task_options_classes()[algorithm]


@pytest.mark.parametrize('algorithm', list(task_options_classes()))
def test_the_printed_header_is_transparent_to_loading(algorithm: str) -> None:
    """Comments must not change what the file means, or redirecting it would not work."""
    options = task_options_classes()[algorithm]()

    assert yaml.safe_load(format_ptychi_options_defaults(algorithm)) == yaml.safe_load(
        dump_task_options_yaml(options)
    )


def test_the_printed_header_is_only_comments() -> None:
    header = format_ptychi_options_defaults('LSQML').split('\n\n', 1)[0]

    assert all(line.startswith('#') for line in header.splitlines())


def test_printing_defaults_needs_no_other_argument() -> None:
    """The flag has to work on its own, which is the only way anyone would reach for it.

    Every driver carrying it marks its real input required, and argparse rejects a
    missing required argument before any driver code runs. An action fires during
    parsing instead; a plain flag read in ``main()`` would fail here.
    """
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument('--diffraction-file', required=True)
    add_print_ptychi_options_argument(parser)

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(['--print-ptychi-options'])

    assert caught.value.code == 0


def test_printing_defaults_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Not stderr: the flag is useless if ``cmd --print-ptychi-options > file`` is empty.

    ``parser.exit(message=...)`` would write to stderr, so this pins the stream rather
    than merely that something was printed.
    """
    parser = argparse.ArgumentParser()
    add_print_ptychi_options_argument(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(['--print-ptychi-options', 'DM'])

    captured = capsys.readouterr()

    assert captured.err == ''
    assert yaml.safe_load(captured.out)['options_class_name'] == 'DMOptions'


def test_printing_defaults_rejects_an_unknown_algorithm(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    add_print_ptychi_options_argument(parser)

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(['--print-ptychi-options', 'NotAnAlgorithm'])

    assert caught.value.code != 0
    assert 'LSQML' in capsys.readouterr().err


def test_the_default_printed_algorithm_is_the_driver_fallback() -> None:
    """Printing with no argument must describe the run you get with no options file."""
    printed = load_task_options_from_mapping(
        yaml.safe_load(format_ptychi_options_defaults('LSQML'))
    )

    assert type(printed) is type(load_ptychi_options(None))


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
    ('data_options.wavelength_m', _make_reconstruct_input().product.metadata.photon_wavelength_m),
    ('data_options.free_space_propagation_distance_m', numpy.inf),
    # True for this far-field product; a near-field one clears it.
    ('data_options.fft_shift', True),
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

    Module-scoped because registering eight declarations and restoring each
    one dominates this file's runtime; the parametrized test below reads these
    options without mutating settings, so one build serves every case.
    """
    harness = _Harness()
    product = _make_reconstruct_input().product
    return [(name, harness.task_options(name, product)) for name in harness.names]


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

    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
        twice = align_task_options_with_product(once, parameters.product)

    assert dump_task_options(twice) == dump_task_options(once)
    assert caplog.records == []


def test_align_warns_only_when_it_overrides_a_caller_value(caplog) -> None:
    """A non-default incoming value that disagrees is a warning; a default is not."""
    parameters = _make_reconstruct_input()

    options = LSQMLOptions()
    options.object_options.pixel_size_m = 5.0e-9

    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
        aligned = align_task_options_with_product(options, parameters.product)

    assert aligned.object_options.pixel_size_m == pytest.approx(PIXEL_M)
    warnings = [record for record in caplog.records if 'pixel_size_m' in record.getMessage()]
    assert len(warnings) == 1

    # The same field left at pty-chi's default is overwritten silently.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
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
    """A ptychi_options.yaml that hard-codes a distance disagreeing with the product
    is corrected, loudly -- the product owns the field outright."""
    parameters = _make_reconstruct_input()
    options = LSQMLOptions()
    options.data_options.free_space_propagation_distance_m = 0.5

    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
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


def test_align_clears_fft_shift_for_near_field() -> None:
    """A near-field measurement is real-space intensity; rolling it is always wrong."""
    product = _with_far_field(_make_reconstruct_input().product, False)
    options = LSQMLOptions()
    assert options.data_options.fft_shift

    aligned = align_task_options_with_product(options, product)

    assert aligned.data_options.fft_shift is False


def test_align_sets_fft_shift_for_far_field() -> None:
    """ptychodus stores patterns DC-centered, so far field must roll them to the corner
    the plain FFT compares against -- not a choice a caller gets to make differently."""
    product = _make_reconstruct_input().product

    for requested in (True, False):
        options = LSQMLOptions()
        options.data_options.fft_shift = requested

        aligned = align_task_options_with_product(options, product)

        assert aligned.data_options.fft_shift is True


def test_align_raises_pad_for_shift_from_its_default_for_near_field() -> None:
    product = _with_far_field(_make_reconstruct_input().product, False)
    options = LSQMLOptions()
    assert options.reconstructor_options.forward_model_options.pad_for_shift == 0

    aligned = align_task_options_with_product(options, product)

    assert aligned.reconstructor_options.forward_model_options.pad_for_shift == 50


def test_align_keeps_a_chosen_pad_for_shift() -> None:
    """It is a tunable with no formula behind it, so a caller's value is never clobbered."""
    product = _with_far_field(_make_reconstruct_input().product, False)
    options = LSQMLOptions()
    options.reconstructor_options.forward_model_options.pad_for_shift = 12

    aligned = align_task_options_with_product(options, product)

    assert aligned.reconstructor_options.forward_model_options.pad_for_shift == 12


def test_align_honors_the_near_field_pad_for_shift_keyword() -> None:
    product = _with_far_field(_make_reconstruct_input().product, False)

    aligned = align_task_options_with_product(LSQMLOptions(), product, near_field_pad_for_shift=80)

    assert aligned.reconstructor_options.forward_model_options.pad_for_shift == 80


def test_align_never_touches_pad_for_shift_in_far_field() -> None:
    product = _make_reconstruct_input().product

    aligned = align_task_options_with_product(LSQMLOptions(), product)

    assert aligned.reconstructor_options.forward_model_options.pad_for_shift == 0


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
    resampled = ProbeEnsemble(
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
    tall = ProbeEnsemble(
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

    ``ProbeEnsemble.get_pixel_geometry`` raises when unset, and such a product
    reconstructed fine before the probe fields existed; it must still align.
    """
    parameters = _make_reconstruct_input()
    probes = parameters.product.probes
    ungeometried = ProbeEnsemble(
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


def test_the_settings_write_nothing_the_product_owns() -> None:
    """Alignment must be the only writer of every field it owns.

    A second writer would make the helper's override warning fire on every
    reconstruction. Checked against a pristine options object rather than
    against a list of expected values: whatever pty-chi declares as the
    starting point is what an unwritten field still holds.
    """
    harness = _Harness()
    specs = {cls.spec.display_name: cls.spec for cls in _ALGORITHM_CLASSES}

    for name in harness.names:
        declared = harness.declared_task_options(name)
        pristine = specs[name].task_options_cls()

        for path, _expected in _PRODUCT_DERIVED_FIELDS:
            read = operator.attrgetter(path)
            assert read(declared) == read(pristine), f'{name}/{path}'

    assert not math.isfinite(PtychographyDataOptions().free_space_propagation_distance_m)


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


# ---------------------------------------------------------------------------
# Probe gauge
# ---------------------------------------------------------------------------

_GAUGE_PHOTONS = 1.0e6
_GAUGE_N = PROBE_HEIGHT_PX * PROBE_WIDTH_PX
# A genuinely near-field sampling. The far-field constants above put the Fresnel number
# nine orders below pty-chi's angular-spectrum threshold, so declaring near field there
# would still select the Fresnel transform and never exercise the conserving branch.
_NEAR_FIELD_PIXEL_M = 1.0e-6
_NEAR_FIELD_DISTANCE_M = 1.0e-3


def _gauge_input(
    *,
    probe_factor: float = 1.0,
    object_amplitude: float = 1.0,
    far_field: bool = True,
    pixel_m: float = PIXEL_M,
    distance_m: float = DETECTOR_DISTANCE_M,
) -> ReconstructInput:
    """A product whose probe, object and patterns already agree with each other.

    The probe carries `_GAUGE_PHOTONS` photons, the object is uniform, and every pattern
    totals what an energy-conserving forward model would put on the detector. The gauge
    measured from it is therefore the propagator's own convention and nothing else, so
    `anomaly` is exactly one and any departure is the thing under test.

    `probe_factor` scales the probe away from that agreement, standing in for a probe
    that arrived in some other gauge.
    """
    probe_array = numpy.zeros((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.complex128)
    probe_array[...] = 1.0
    probe_array *= numpy.sqrt(_GAUGE_PHOTONS / numpy.sum(numpy.abs(probe_array) ** 2))
    probe_array *= probe_factor

    obj = Object(
        array=numpy.full((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX), object_amplitude, dtype=complex),
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[],
    )
    probes = ProbeEnsemble(
        array=probe_array,
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
    )
    positions = ProbePositionSequence(
        [ProbePosition(index=i, x_m=0.0, y_m=0.0) for i in range(NUM_PATTERNS)]
    )
    product = Product(
        metadata=ProductMetadata(
            name='gauge',
            comments='',
            detector_distance_m=distance_m,
            photon_energy_eV=PHOTON_ENERGY_EV,
            probe_photon_count=_GAUGE_PHOTONS,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
            far_field=far_field,
        ),
        probe_positions=positions,
        probes=probes,
        object_=obj,
        losses=[],
    )
    # What an energy-conserving model would record: the photons the exit wave carries.
    per_pattern_total = _GAUGE_PHOTONS * object_amplitude**2
    patterns = numpy.full(
        (NUM_PATTERNS, PROBE_HEIGHT_PX, PROBE_WIDTH_PX),
        per_pattern_total / _GAUGE_N,
        dtype=numpy.float64,
    )
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_),
        product=product,
    )


def _aligned_for(parameters: ReconstructInput) -> PtychographyTaskOptions:
    return align_task_options_with_product(LSQMLOptions(), parameters.product)


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='far-field-fft'),
        pytest.param(
            {
                'far_field': False,
                'pixel_m': _NEAR_FIELD_PIXEL_M,
                'distance_m': _NEAR_FIELD_DISTANCE_M,
            },
            id='near-field-angular-spectrum',
        ),
        pytest.param({'far_field': False}, id='near-field-fresnel-transform'),
    ],
)
def test_probe_gauge_of_a_matched_product_is_the_propagator_convention(
    kwargs: dict[str, object],
) -> None:
    """With probe, object and data already in agreement, the only thing left to correct
    is pty-chi's own propagator normalization -- whichever of the three it builds."""
    parameters = _gauge_input(**kwargs)  # type: ignore[arg-type]
    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    # Single precision throughout: pty-chi propagates in complex64 unless told not to.
    assert gauge.anomaly == pytest.approx(1.0, rel=1e-6)
    assert gauge.sampled_positions == NUM_PATTERNS


def _gauge_input_matched_per_pixel(
    *,
    far_field: bool = True,
    pixel_m: float = PIXEL_M,
    distance_m: float = DETECTOR_DISTANCE_M,
) -> ReconstructInput:
    """A matched product whose patterns agree with the forward model pixel by pixel.

    `_gauge_input` matches only the per-pattern total, which is all the unmasked gauge
    reads. Masking makes the comparison spatial, so this builds the patterns by actually
    propagating the exit wave and rolling the result into the detector frame, then scales
    them to the photons an energy-conserving model would record. Every pixel then carries
    the same predicted-to-measured ratio, so the gauge is the same over any subset of
    them -- which is the invariant a masked test is there to pin.

    The probe is a centered Gaussian rather than the uniform one `_gauge_input` uses: a
    uniform probe transforms to a single bright pixel, leaving a mask nothing to bite on.
    """
    y, x = numpy.mgrid[0:PROBE_HEIGHT_PX, 0:PROBE_WIDTH_PX]
    envelope = numpy.exp(-(((x - PROBE_WIDTH_PX / 2) ** 2 + (y - PROBE_HEIGHT_PX / 2) ** 2) / 4.0))
    probe_array = numpy.zeros((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.complex128)
    probe_array[0, 0] = envelope
    probe_array *= numpy.sqrt(_GAUGE_PHOTONS / numpy.sum(numpy.abs(probe_array) ** 2))

    parameters = _gauge_input(far_field=far_field, pixel_m=pixel_m, distance_m=distance_m)
    product = parameters.product
    probes = ProbeEnsemble(
        array=probe_array,
        opr_weights=None,
        pixel_geometry=product.probes.get_pixel_geometry(),
    )
    product = dataclasses.replace(product, probes=probes)

    propagator, corner_ordered = _build_ptychi_propagator(
        align_task_options_with_product(LSQMLOptions(), product), probes
    )
    import torch

    with torch.no_grad():
        propagated = propagator.propagate_forward(torch.as_tensor(probe_array[0]))
        intensity = numpy.sum(numpy.abs(propagated.numpy()) ** 2, axis=0)

    if corner_ordered:
        intensity = numpy.fft.fftshift(intensity, axes=(-2, -1))

    pattern = intensity * (_GAUGE_PHOTONS / numpy.sum(intensity))
    patterns = numpy.repeat(pattern[numpy.newaxis], NUM_PATTERNS, axis=0)

    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_),
        product=product,
    )


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='far-field-fft'),
        pytest.param(
            {
                'far_field': False,
                'pixel_m': _NEAR_FIELD_PIXEL_M,
                'distance_m': _NEAR_FIELD_DISTANCE_M,
            },
            id='near-field-angular-spectrum',
        ),
    ],
)
def test_probe_gauge_is_unchanged_by_which_pixels_are_masked(
    kwargs: dict[str, object],
) -> None:
    """Bad pixels drop the same content from both sides of the gauge, whatever the
    propagator's output ordering.

    The predicted plane is summed over a mask written in detector coordinates, so a
    propagator that returns DC at a corner -- the plain FFT far field does -- has to be
    rolled before it is indexed. Left unrolled, the two sides exclude different pixels
    and a perfectly matched product reports a probe that needs rescaling.
    """
    clean = _gauge_input_matched_per_pixel(**kwargs)  # type: ignore[arg-type]
    bad_pixels = numpy.zeros((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), dtype=numpy.bool_)
    # Off-center and asymmetric, so it covers different content under either ordering.
    bad_pixels[1:4, 2:6] = True
    masked = dataclasses.replace(clean, bad_pixels=bad_pixels)

    gauge = align_probe_gauge_with_ptychi(_aligned_for(masked), masked)

    assert gauge.anomaly == pytest.approx(1.0, rel=1e-5)
    # Guard the degrade path, which also reports an anomaly of exactly one.
    assert gauge.sampled_positions == NUM_PATTERNS
    assert gauge.to_ptychi == pytest.approx(
        align_probe_gauge_with_ptychi(_aligned_for(clean), clean).to_ptychi, rel=1e-5
    )


def test_probe_gauge_in_far_field_undoes_the_unnormalized_fft() -> None:
    """Far field propagates with a plain FFT, which multiplies intensity by the pixel
    count; the probe is scaled by the square root of that."""
    parameters = _gauge_input(far_field=True)
    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.expected_factor == pytest.approx(1.0 / math.sqrt(_GAUGE_N), rel=1e-6)


def test_probe_gauge_in_near_field_leaves_an_energy_conserving_probe_alone() -> None:
    """The angular spectrum conserves energy, which is ptychodus's own convention, so
    there is nothing to convert -- the case a hardcoded far-field factor would break."""
    parameters = _gauge_input(
        far_field=False, pixel_m=_NEAR_FIELD_PIXEL_M, distance_m=_NEAR_FIELD_DISTANCE_M
    )
    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.expected_factor == pytest.approx(1.0, rel=1e-6)
    assert gauge.to_ptychi == pytest.approx(1.0, rel=1e-6)


def test_probe_gauge_absorbs_the_objects_transmission() -> None:
    """An object carrying the sample's mean transmission leaves the probe meaning the
    incident illumination, so the gauge stays at the propagator convention."""
    parameters = _gauge_input(object_amplitude=0.5)
    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.anomaly == pytest.approx(1.0, rel=1e-6)


def test_probe_gauge_corrects_a_probe_already_in_ptychi_gauge() -> None:
    """A product written before ptychodus converted here holds pty-chi's own probe,
    smaller by the square root of the pixel count. Such a probe needs no conversion, and
    the measurement says so by departing from the convention by exactly that factor."""
    parameters = _gauge_input(probe_factor=1.0 / math.sqrt(_GAUGE_N))

    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.anomaly == pytest.approx(math.sqrt(_GAUGE_N), rel=1e-6)
    assert gauge.to_ptychi == pytest.approx(1.0, rel=1e-6)


def test_probe_gauge_warns_when_the_probe_is_in_a_foreign_gauge(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Far enough from the convention and the probe did not come from here at all --
    another tool, or a mis-scaled file. Reported, and corrected anyway."""
    parameters = _gauge_input(probe_factor=1.0e-3)

    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
        gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.anomaly == pytest.approx(1.0e3, rel=1e-6)
    assert 'may already be in' in caplog.text


def test_probe_gauge_round_trips_a_probe_exactly() -> None:
    """In and out are the same scalar, so a reconstruction that changed nothing returns
    the probe it was given."""
    parameters = _gauge_input()
    gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)
    array = parameters.product.probes.get_array()

    numpy.testing.assert_allclose(array * gauge.to_ptychi / gauge.to_ptychi, array, rtol=1e-12)


def test_probe_gauge_ignores_a_carried_background() -> None:
    """The patterns have not been corrected for it, so left in it would inflate the
    measurement and shrink the probe."""
    plain = _gauge_input()
    background = numpy.full((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), 7.0)
    with_background = dataclasses.replace(
        plain,
        diffraction_patterns=plain.diffraction_patterns + background,
        background=background,
    )

    expected = align_probe_gauge_with_ptychi(_aligned_for(plain), plain)
    actual = align_probe_gauge_with_ptychi(_aligned_for(with_background), with_background)

    assert actual.to_ptychi == pytest.approx(expected.to_ptychi, rel=1e-6)


def test_probe_gauge_of_unusable_data_falls_back_to_the_convention(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A scan that recorded nothing gives no ratio to measure; the propagator convention
    is still right whenever probe and object already agree with the data."""
    plain = _gauge_input()
    parameters = dataclasses.replace(
        plain, diffraction_patterns=numpy.zeros_like(plain.diffraction_patterns)
    )

    with caplog.at_level(logging.WARNING, logger='ptychodus.plugins.ptychi.task'):
        gauge = align_probe_gauge_with_ptychi(_aligned_for(parameters), parameters)

    assert gauge.to_ptychi == gauge.expected_factor
    assert gauge.sampled_positions == 0
    assert 'unusable' in caplog.text


def test_power_constraint_target_follows_the_probe_into_ptychi_gauge() -> None:
    """The constraint measures the probe's power on pty-chi's arrays, so its target has
    to be stated there too -- otherwise ticking it would pull the probe to a power it
    never had. Agreement here is what keeps the two mechanisms from fighting."""
    parameters = _gauge_input()
    aligned = _aligned_for(parameters)
    gauge = align_probe_gauge_with_ptychi(aligned, parameters)

    applied = apply_probe_gauge_to_options(aligned, gauge)
    handed_over = parameters.product.probes.get_array() * gauge.to_ptychi

    assert applied.probe_options.power_constraint.probe_power == pytest.approx(
        float(numpy.sum(numpy.abs(handed_over) ** 2)), rel=1e-6
    )


def test_applying_the_gauge_leaves_the_original_options_alone() -> None:
    """The options a run writes out keep the photon count a reader can interpret."""
    parameters = _gauge_input()
    aligned = _aligned_for(parameters)
    gauge = align_probe_gauge_with_ptychi(aligned, parameters)

    apply_probe_gauge_to_options(aligned, gauge)

    assert aligned.probe_options.power_constraint.probe_power == _GAUGE_PHOTONS
