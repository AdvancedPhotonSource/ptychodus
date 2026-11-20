"""The shared standard-reconstruction pipeline and the profiles the drivers declare.

The parser tests matter because ten beamline commands now get their entire command-line
surface from one builder: a mistake in a profile field is a silent change to a published
CLI, and nothing else would catch it. The filter tests cover the only arithmetic in the
module.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import logging

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.assemble import AssembledDiffractionData  # noqa: E402
from ptychodus.api.diffraction import (  # noqa: E402
    DiffractionIndexes,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence  # noqa: E402
from ptychodus.cli import _reconstruct_standard as rs  # noqa: E402
from ptychodus.cli._reconstruct_common import invalid_count_threshold  # noqa: E402
from ptychodus.cli._reconstruct_standard import InstrumentProfile
from ptychodus.plugins.aps31id_lamni.diffraction_file import LamNIDiffractionFileReader
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.probe import Probe, ProbeEnsemble, ProbeGeometry
from ptychodus.api.accumulate import IncidentBeamEstimate, compute_diffraction_moments
from ptychodus.api.object import Object, ObjectGeometry, compute_object_geometry
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.scattering.geometry import ScatteringGeometry
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.propagate import compute_magnification

DETECTOR_PIXEL_M = 7.5e-5
# noqa: E402

from diffraction_dataset_fixtures import FailingArray, make_dataset  # noqa: E402

logger = logging.getLogger('test_reconstruct_standard')


def _profiles() -> dict[str, InstrumentProfile]:
    import importlib

    names = (
        '12ide',
        '26idc',
        '2ide',
        '34ide',
        'bnp',
        'isn',
        'lamni',
        'velociprobe',
        'foldslice',
        'polar',
    )
    return {n: importlib.import_module(f'ptychodus.cli.reconstruct_{n}').PROFILE for n in names}


def _defaults(profile: InstrumentProfile) -> argparse.Namespace:
    parser = rs.build_standard_parser(profile)
    return parser.parse_args(
        ['--diffraction-file', 'd', '--position-file', 'p', '--output-directory', 'o']
    )


def test_every_profile_builds_a_parser() -> None:
    for name, profile in _profiles().items():
        assert rs.build_standard_parser(profile).parse_known_args(
            ['--diffraction-file', 'd', '--position-file', 'p', '--output-directory', 'o']
        ), name


def test_every_profile_can_print_option_defaults() -> None:
    """All ten beamline commands must offer the flag, and offer it without any input.

    It is how a user obtains the document ``--options`` takes, so a driver missing it
    leaves no way to start one but finishing a reconstruction first.
    """
    for name, profile in _profiles().items():
        parser = rs.build_standard_parser(profile)

        with pytest.raises(SystemExit) as caught:
            parser.parse_args(['--print-options'])

        assert caught.value.code == 0, name


def test_instrument_defaults_are_carried_from_the_profile() -> None:
    profiles = _profiles()

    assert _defaults(profiles['12ide']).detector_distance_m is None
    assert profiles['12ide'].default_detector_distance_m == 10.2
    assert profiles['bnp'].default_detector_distance_m == 2.06
    assert profiles['isn'].default_detector_distance_m == 6.16

    # 34-ID-E is the one instrument with its own bad-pixel reader and a pixel-size fallback.
    assert _defaults(profiles['34ide']).bad_pixels_file_type == 'APS_34IDE_Bad_Pixels'
    assert _defaults(profiles['12ide']).bad_pixels_file_type == 'NPY_Bad_Pixels'
    assert profiles['34ide'].default_detector_pixel_size_m == 75e-6

    # velociprobe is the one instrument whose two readers differ.
    assert profiles['velociprobe'].diffraction_reader == 'APS_Velociprobe'
    assert profiles['velociprobe'].position_reader == 'APS_Velociprobe_PE'


def test_no_profile_presets_a_defocus() -> None:
    """The focus is stated or estimated, so every profile leaves it unset."""
    for name, profile in _profiles().items():
        args = _defaults(profile)

        assert args.focus_object_distance_m is None, name
        assert not hasattr(args, 'fzp_defocus_m'), name
        assert not hasattr(profile, 'default_fzp_defocus_m'), name
        assert (
            args.parallax_bin_size_px,
            args.parallax_min_significance,
            args.parallax_min_defocus_dof,
        ) == (4, 3.0, 1.0), name


@pytest.mark.parametrize('flag', ['--fzp-defocus-m', '--estimate-defocus'])
def test_the_retired_defocus_flags_are_refused(flag: str) -> None:
    parser = rs.build_standard_parser(_profiles()['lamni'])

    with pytest.raises(SystemExit):
        parser.parse_args(
            ['--diffraction-file', 'd', '--position-file', 'p', '--output-directory', 'o', flag]
            + ([] if flag == '--estimate-defocus' else ['8e-4'])
        )


def test_lamni_marks_invalid_uint16_pixels_with_its_12_bit_maximum() -> None:
    """Its fly scans store 12-bit Eiger frames as uint16; uint32 frames count past 4095."""
    invalid_counts = _profiles()['lamni'].default_invalid_counts

    assert invalid_count_threshold(numpy.dtype('uint16'), invalid_counts) == 4095
    assert invalid_count_threshold(numpy.dtype('uint32'), invalid_counts) == (
        numpy.iinfo(numpy.uint32).max
    )


def test_every_lamni_path_reads_one_preset() -> None:
    """The batch drivers and PEAR's lynx entries pass the reader's constant, as this does."""
    assert _profiles()['lamni'].default_invalid_counts is LamNIDiffractionFileReader.INVALID_COUNTS


def test_other_profiles_mark_invalid_pixels_with_the_dtype_maximum() -> None:
    for name, profile in _profiles().items():
        if name != 'lamni':
            assert not profile.default_invalid_counts, name


def test_the_crop_extent_is_left_unset_by_default_and_accepts_full() -> None:
    """Unset is resolved later: the --probe-file extent, else auto."""
    profile = _profiles()['12ide']
    assert _defaults(profile).crop_extent_px is None

    parser = rs.build_standard_parser(profile)
    args = parser.parse_args(
        ['--diffraction-file', 'd', '--position-file', 'p', '--output-directory', 'o']
        + ['--crop-extent-px', 'full']
    )
    assert args.crop_extent_px == 'full'


def test_unconditioned_profile_drops_the_conditioning_options() -> None:
    args = _defaults(_profiles()['foldslice'])

    for absent in (
        'crop_extent_px',
        'beam_center_x_px',
        'beam_center_y_px',
        'min_detector_sum',
        'max_valid_count',
        'bad_pixels_file',
        'bad_pixels_file_type',
        'background_file',
        'background_file_type',
        'subtract_background',
        'flat_field_file',
        'flat_field_file_type',
        'divide_flat_field',
    ):
        assert not hasattr(args, absent), absent

    # and gains the two it needs instead
    assert args.detector_pixel_size_m is None
    assert args.probe_photon_count is None


def test_pattern_filters_are_offered_only_where_the_profile_asks() -> None:
    profiles = _profiles()
    polar = _defaults(profiles['polar'])

    assert polar.dp_mad_k is None
    assert polar.i0_mad_k is None
    assert polar.drop_leading_patterns == 0
    assert polar.drop_trailing_patterns == 0

    for name in ('12ide', '26idc', '2ide', '34ide', 'bnp', 'isn', 'lamni', 'velociprobe'):
        args = _defaults(profiles[name])
        for absent in ('dp_mad_k', 'i0_mad_k', 'drop_leading_patterns', 'drop_trailing_patterns'):
            assert not hasattr(args, absent), f'{name}.{absent}'


def test_diffraction_reader_follows_the_file_suffix() -> None:
    from pathlib import Path

    profile = _profiles()['26idc']

    assert profile.get_diffraction_reader(Path('scan_29_img_005604.tif')) == 'CNM_APS_HXN_TIFF'
    assert profile.get_diffraction_reader(Path('scan_126_img_075988.tif.gz')) == 'CNM_APS_HXN_TIFF'
    assert profile.get_diffraction_reader(Path('scan_43_000099.h5')) == 'CNM_APS_HXN'
    # A profile without a suffix table always uses its one reader.
    isn = _profiles()['isn']
    assert isn.get_diffraction_reader(Path('anything.tif')) == isn.diffraction_reader


def test_no_profile_presets_a_zone_plate() -> None:
    """Every driver generates its initial probe from the mean pattern unless told otherwise."""
    for name, profile in _profiles().items():
        assert _defaults(profile).fzp_preset == '', name
        assert not hasattr(profile, 'default_fzp_preset'), name


def test_every_profile_generates_the_initial_object_by_stxm_dpc() -> None:
    for name, profile in _profiles().items():
        assert _defaults(profile).initial_object is rs.InitialObject.STXM_DPC, name

    parser = rs.build_standard_parser(_profiles()['bnp'])
    base = ['--diffraction-file', 'd', '--position-file', 'p', '--output-directory', 'o']

    assert parser.parse_args([*base, '--initial-object', 'uniform']).initial_object is (
        rs.InitialObject.UNIFORM
    )

    with pytest.raises(SystemExit):
        parser.parse_args([*base, '--initial-object', 'random'])


# --- the initial object ----------------------------------------------------------------


_TRANSMISSION = 0.9


def _phase_scan() -> tuple[
    ObjectGeometry, IncidentBeamEstimate, ProbePositionSequence, Probe, ImageExtent
]:
    """A smooth phase object under a defocused disk probe, its far-field patterns, and
    the incident beam they imply with the transmission fixed."""
    pixel_size_m = 50e-9
    pixel_geometry = PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m)
    probe_px = 32
    yy, xx = numpy.indices((probe_px, probe_px)) - (probe_px - 1) / 2
    probe = Probe(
        array=(numpy.hypot(xx, yy) < 7).astype(complex) * numpy.exp(0.02j * (xx**2 + yy**2)),
        pixel_geometry=pixel_geometry,
    )
    positions = ProbePositionSequence(
        [
            ProbePosition(
                index=iy * 7 + ix,
                x_m=(ix - 3) * 4 * pixel_size_m,
                y_m=(iy - 3) * 4 * pixel_size_m,
            )
            for iy in range(7)
            for ix in range(7)
        ]
    )
    object_geometry = compute_object_geometry(
        positions,
        ProbeGeometry(
            width_px=probe_px,
            height_px=probe_px,
            pixel_width_m=pixel_size_m,
            pixel_height_m=pixel_size_m,
        ),
        padding_px=4,
    )
    oy, ox = numpy.indices((object_geometry.height_px, object_geometry.width_px))
    phase = numpy.sin(ox / 6.0) * numpy.cos(oy / 9.0)
    product = Product(
        metadata=ProductMetadata(
            name='phase',
            comments='',
            detector_distance_m=2.0,
            photon_energy_eV=10_000.0,
            probe_photon_count=1.0,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=positions,
        probes=ProbeEnsemble.from_probe(probe),
        object_=Object(
            array=numpy.exp(1j * phase)[numpy.newaxis],
            pixel_geometry=pixel_geometry,
            center=object_geometry.get_center(),
        ),
        losses=[],
    )
    data = generate_diffraction_data(product, None)
    moments = compute_diffraction_moments(
        data.get_indexes(), data.get_patterns(), bad_pixels=data.get_bad_pixels()
    )
    assert moments is not None
    incident_beam = IncidentBeamEstimate(
        moments=moments,
        incident_photon_count=float(moments.detector_sums.max()),
        mean_amplitude_transmission=_TRANSMISSION,
        decay_corrected=False,
    )
    return object_geometry, incident_beam, positions, probe, data.get_image_extent()


def _generate(
    initial_object: rs.InitialObject,
    *,
    far_field: bool = True,
    scattering_geometry: ScatteringGeometry | None = None,
    incident_beam: IncidentBeamEstimate | None = None,
) -> numpy.ndarray:
    object_geometry, beam, positions, probe, extent = _phase_scan()
    object_ = rs._generate_initial_object(
        logger,
        initial_object,
        object_geometry,
        beam if incident_beam is None else incident_beam,
        positions,
        probe,
        detector_extent=extent,
        far_field=far_field,
        scattering_geometry=(
            ScatteringGeometry() if scattering_geometry is None else scattering_geometry
        ),
    )
    return object_.get_array()[0]


def test_stxm_dpc_generates_a_structured_object_at_the_measured_transmission() -> None:
    array = _generate(rs.InitialObject.STXM_DPC)

    assert numpy.std(numpy.angle(array)) > 0.05
    assert numpy.mean(numpy.abs(array)) == pytest.approx(_TRANSMISSION, rel=0.1)


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(initial_object=rs.InitialObject.UNIFORM),
        dict(initial_object=rs.InitialObject.STXM_DPC, far_field=False),
        dict(
            initial_object=rs.InitialObject.STXM_DPC,
            scattering_geometry=ScatteringGeometry(two_theta_deg=30.0),
        ),
    ],
)
def test_otherwise_the_initial_object_is_uniform_at_the_transmission(kwargs: dict) -> None:
    numpy.testing.assert_allclose(_generate(**kwargs), _TRANSMISSION)


def test_a_failed_stxm_dpc_falls_back_to_uniform(caplog: pytest.LogCaptureFixture) -> None:
    _, beam, *_ = _phase_scan()
    # No signal anywhere: STXM-DPC refuses a mean intensity of zero.
    empty = replace(
        beam,
        moments=replace(beam.moments, detector_sums=numpy.zeros_like(beam.moments.detector_sums)),
    )

    with caplog.at_level(logging.WARNING):
        array = _generate(rs.InitialObject.STXM_DPC, incident_beam=empty)

    assert 'STXM-DPC failed' in caplog.text
    numpy.testing.assert_allclose(array, _TRANSMISSION)


def test_logger_names_stay_per_driver() -> None:
    """Log provenance is what makes a beamline log readable; the shared body must not eat it."""
    for name, profile in _profiles().items():
        assert profile.logger_name == f'reconstruct_{name}'


def test_every_profile_offers_prepare_only() -> None:
    for name, profile in _profiles().items():
        assert _defaults(profile).prepare_only is False, name


def test_prepare_only_and_dry_run_are_exclusive() -> None:
    parser = rs.build_standard_parser(_profiles()['2ide'])

    with pytest.raises(SystemExit) as caught:
        parser.parse_args(
            [
                '--diffraction-file',
                'd',
                '--position-file',
                'p',
                '--output-directory',
                'o',
                '--prepare-only',
                '--dry-run',
            ]
        )

    assert caught.value.code == 2


def test_prepare_only_refuses_to_skip_the_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    # A prepared directory without diffraction.h5 cannot be reconstructed, so the
    # combination is rejected before any file is read.
    monkeypatch.setattr(
        'sys.argv',
        [
            'ptychodus-reconstruct-2ide',
            '--diffraction-file',
            'd',
            '--position-file',
            'p',
            '--output-directory',
            'o',
            '--prepare-only',
            '--no-save-diffraction',
        ],
    )

    with pytest.raises(SystemExit) as caught:
        rs.run_standard_reconstruction(_profiles()['2ide'])

    assert caught.value.code == 2


# --- pattern filters ------------------------------------------------------------------


def test_mad_bounds_keeps_every_positive_point_of_a_constant_trace() -> None:
    """Zero MAD is degenerate: the interval opens upward rather than rejecting everything."""
    lower, upper = rs._mad_bounds(numpy.full(32, 7.0), 5.0)
    assert lower <= 7.0 <= upper


def test_mad_bounds_brackets_the_median_of_a_spread() -> None:
    values = numpy.array([10.0, 11.0, 9.0, 10.5, 9.5, 1000.0])
    lower, upper = rs._mad_bounds(values, 5.0)
    assert lower <= 10.0 <= upper
    assert upper < 1000.0


def _assembled(indexes: DiffractionIndexes) -> AssembledDiffractionData:
    """A real AssembledDiffractionData over constant 2x2 frames.

    The filters under test slice one of these and rebuild it, so they are handed
    the real class rather than a stand-in: a stand-in only has to look right, and
    nothing would notice it drifting from the class it imitates.
    """
    return AssembledDiffractionData(
        indexes=indexes,
        patterns=numpy.ones((indexes.size, 2, 2), dtype=numpy.uint16),
        pixel_geometry=PixelGeometry(width_m=1.0, height_m=1.0),
        bad_pixels=numpy.zeros((2, 2), dtype=numpy.bool_),
    )


def _never_read_dataset() -> SimpleDiffractionDataset:
    """A dataset whose only array raises if anything reads it.

    The two bounds tests below assert the raw dataset goes untouched when the MAD
    bound is off. A dataset that fails loudly proves that; the `None` they used to
    pass only proved that nothing was type-checking them.
    """
    return make_dataset(
        [FailingArray('never-read', AssertionError('raw_dataset must not be read'))],
        (1, 1),
        pattern_dtype=numpy.uint16,
    )


def test_select_patterns_returns_the_input_when_nothing_is_dropped() -> None:
    data = _assembled(numpy.arange(4))
    keep = numpy.ones(4, dtype=bool)
    assert rs._select_patterns(logger, data, keep, 'A filter') is data


def test_select_patterns_refuses_to_drop_everything() -> None:
    data = _assembled(numpy.arange(4))
    keep = numpy.zeros(4, dtype=bool)

    with pytest.raises(ValueError, match='dropped every pattern'):
        rs._select_patterns(logger, data, keep, 'A filter')


def test_trim_ends_refuses_to_consume_every_pattern() -> None:
    data = _assembled(numpy.arange(4))

    with pytest.raises(ValueError, match='consume all 4'):
        rs._trim_ends(logger, data, 2, 2)


def test_reject_i0_outliers_is_a_no_op_without_i0() -> None:
    data = _assembled(numpy.arange(4))
    positions = ProbePositionSequence([ProbePosition(i, float(i), 0.0) for i in range(4)])

    assert positions.get_i0() is None
    assert rs._reject_i0_outliers(logger, data, positions, 5.0) is data


def test_detector_sum_bounds_passes_min_through_when_mad_is_off() -> None:
    args = argparse.Namespace(min_detector_sum=42, dp_mad_k=None)
    assert rs._detector_sum_bounds(logger, args, _never_read_dataset(), None, None, None) == (
        42,
        None,
    )


def test_detector_sum_bounds_is_inert_when_both_options_are_off() -> None:
    args = argparse.Namespace(min_detector_sum=None, dp_mad_k=None)
    assert rs._detector_sum_bounds(logger, args, _never_read_dataset(), None, None, None) == (
        None,
        None,
    )


def test_select_patterns_keeps_the_detector_maps() -> None:
    data = AssembledDiffractionData(
        indexes=numpy.arange(3),
        patterns=numpy.ones((3, 2, 2)),
        pixel_geometry=PixelGeometry(width_m=1.0, height_m=1.0),
        bad_pixels=numpy.zeros((2, 2), dtype=numpy.bool_),
        background=numpy.full((2, 2), 4.0),
        flat_field=numpy.full((2, 2), 0.5),
    )

    selected = rs._select_patterns(logger, data, numpy.array([True, False, True]), 'A filter')

    numpy.testing.assert_array_equal(selected.get_background(), 4.0)
    numpy.testing.assert_array_equal(selected.get_flat_field(), 0.5)


def test_conditioning_profiles_offer_the_detector_map_options() -> None:
    args = _defaults(_profiles()['12ide'])

    assert args.background_file is None
    assert args.background_file_type == 'NPY'
    assert not args.subtract_background
    assert args.flat_field_file is None
    assert args.flat_field_file_type == 'NPY'
    assert not args.divide_flat_field


class TestNearFieldOptIn:
    """Near field is declared, never inferred, and no existing run changes."""

    def test_every_profile_defaults_to_far_field(self) -> None:
        for name, profile in _profiles().items():
            assert profile.default_far_field is True, name

    def test_the_flag_is_unset_rather_than_false_by_default(self) -> None:
        # The file may declare the regime, so "the user said nothing" has to stay
        # distinguishable from "the user asked for far field".
        for name, profile in _profiles().items():
            assert _defaults(profile).near_field is None, name

    def test_the_flag_is_accepted_with_a_focus_distance(self) -> None:
        parser = rs.build_standard_parser(_profiles()['lamni'])

        args = parser.parse_args(
            [
                '--diffraction-file',
                'd',
                '--position-file',
                'p',
                '--output-directory',
                'o',
                '--near-field',
                '--focus-object-distance-m=-9e-4',
            ]
        )

        assert args.near_field is True
        assert args.focus_object_distance_m == pytest.approx(-9e-4)

    def test_a_negative_focus_distance_needs_the_equals_form(self) -> None:
        """The common case is negative, and argparse reads a bare -9e-4 as an option.

        Its negative-number matcher only admits plain decimals, so scientific notation
        has to be attached with '='. Pinned because the help text promises it.
        """
        parser = rs.build_standard_parser(_profiles()['lamni'])
        base = [
            '--diffraction-file',
            'd',
            '--position-file',
            'p',
            '--output-directory',
            'o',
        ]

        assert parser.parse_args(
            [*base, '--focus-object-distance-m', '-0.0009']
        ).focus_object_distance_m == pytest.approx(-9e-4)

        with pytest.raises(SystemExit):
            parser.parse_args([*base, '--focus-object-distance-m', '-9e-4'])

    def test_the_probe_propagation_distance_defaults_to_zero(self) -> None:
        for name, profile in _profiles().items():
            assert _defaults(profile).probe_propagation_distance_m == 0.0, name


class TestNearFieldProbeGeometry:
    """The regime decides how the detector grid maps onto the object."""

    def _geometry(self, *, magnification: float) -> ProbeGeometry:
        return ProbeGeometry.from_near_field(
            PixelGeometry(width_m=DETECTOR_PIXEL_M, height_m=DETECTOR_PIXEL_M),
            ImageExtent(width_px=64, height_px=64),
            magnification=magnification,
        )

    def test_a_parallel_beam_samples_at_the_detector_pitch(self) -> None:
        # No focusing optic means M = 1, where the angular spectrum preserves pitch.
        geometry = self._geometry(magnification=compute_magnification(1.0, 0.0))

        assert geometry.pixel_width_m == pytest.approx(DETECTOR_PIXEL_M)

    def test_a_cone_beam_demagnifies_the_detector_pitch(self) -> None:
        magnification = compute_magnification(1.0, -1.0e-3)
        geometry = self._geometry(magnification=magnification)

        assert magnification == pytest.approx(1001.0)
        assert geometry.pixel_width_m == pytest.approx(DETECTOR_PIXEL_M / magnification)

    def test_an_upstream_focus_magnifies_more_than_a_downstream_one(self) -> None:
        """The sign is a real geometry choice, not a magnitude to take the absolute of."""
        upstream = compute_magnification(1.0, -1.0e-3)
        downstream = compute_magnification(1.0, +1.0e-3)

        assert upstream > downstream


# --- scattering geometry ---------------------------------------------------------------


class TestScatteringGeometry:
    """Only the off-axis instruments offer the angles; elsewhere the geometry is forward."""

    def test_only_26idc_and_34ide_offer_the_angles(self) -> None:
        for name, profile in _profiles().items():
            args = _defaults(profile)
            offered = name in ('26idc', '34ide')

            assert profile.offers_scattering_geometry is offered, name
            assert hasattr(args, 'two_theta_deg') is offered, name

    def test_the_command_line_wins_over_the_file(self) -> None:
        from ptychodus.api.scattering.geometry import ScatteringGeometry

        args = _defaults(_profiles()['26idc'])
        args.two_theta_deg = 47.35
        recorded = ScatteringGeometry(two_theta_deg=10.0, incidence_angle_deg=30.0)

        geometry = rs._resolve_scattering_geometry(logger, args, recorded, _profiles()['26idc'])

        assert geometry == ScatteringGeometry(two_theta_deg=47.35, incidence_angle_deg=30.0)

    def test_nothing_said_is_forward(self) -> None:
        from ptychodus.api.scattering.geometry import ScatteringGeometry

        args = _defaults(_profiles()['34ide'])

        assert (
            rs._resolve_scattering_geometry(logger, args, None, _profiles()['34ide'])
            == ScatteringGeometry()
        )

    def test_a_profile_without_the_switch_ignores_the_file(self) -> None:
        from ptychodus.api.scattering.geometry import ScatteringGeometry

        args = _defaults(_profiles()['2ide'])
        recorded = ScatteringGeometry(two_theta_deg=10.0)

        assert (
            rs._resolve_scattering_geometry(logger, args, recorded, _profiles()['2ide'])
            == ScatteringGeometry()
        )

    def test_prepare_only_writes_exit_frame_positions_and_the_geometry(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from types import SimpleNamespace

        from ptychodus.api.scattering.frame import compute_exit_frame_transform
        from ptychodus.api.diffraction import SimpleDiffractionArray
        from ptychodus.api.io import StandardFileLayout, load_product
        from ptychodus.api.scattering.geometry import ScatteringGeometry

        rng = numpy.random.default_rng(0)
        num_patterns = 16
        indexes = numpy.arange(num_patterns)
        patterns = rng.poisson(50.0, size=(num_patterns, 64, 64)).astype(numpy.uint16)
        dataset = make_dataset(
            [SimpleDiffractionArray('scan', indexes, patterns)],
            (64, 64),
            pixel_geometry=PixelGeometry(width_m=DETECTOR_PIXEL_M, height_m=DETECTOR_PIXEL_M),
        )
        lab_positions = ProbePositionSequence(
            [ProbePosition(index=int(i), x_m=(i % 4) * 5e-8, y_m=(i // 4) * 5e-8) for i in indexes]
        )

        def reader(result: object) -> SimpleNamespace:
            return SimpleNamespace(read=lambda path: result)

        registry = SimpleNamespace(
            diffraction_file_readers=SimpleNamespace(
                get_strategy_by_name=lambda name: reader(dataset)
            ),
            probe_position_file_readers=SimpleNamespace(
                get_strategy_by_name=lambda name: reader(lab_positions)
            ),
            reconstructor_libraries=rs.PluginRegistry.load_plugins().reconstructor_libraries,
        )
        monkeypatch.setattr(rs.PluginRegistry, 'load_plugins', lambda: registry)
        output = tmp_path / 'prepared'
        monkeypatch.setattr(
            'sys.argv',
            [
                'ptychodus-reconstruct-26idc',
                '--diffraction-file',
                str(tmp_path / 'scan_1_img_000000.tif'),
                '--position-file',
                str(tmp_path / '26idbSOFT_0001.mda'),
                '--output-directory',
                str(output),
                '--photon-energy-eV',
                '11200',
                '--detector-distance-m',
                '0.89',
                '--two-theta-deg',
                '47.35',
                '--detector-gamma-deg',
                '-0.27',
                '--incidence-angle-deg',
                '24.15',
                '--prepare-only',
            ],
        )

        assert rs.run_standard_reconstruction(_profiles()['26idc']) == 0

        product = load_product(StandardFileLayout.PRODUCT.path(output))
        geometry = ScatteringGeometry(47.35, -0.27, 24.15)
        transform = compute_exit_frame_transform(geometry)

        assert product.metadata.scattering_geometry == geometry

        for stored, lab in zip(product.probe_positions, lab_positions):
            expected = transform(lab)
            assert stored.index == lab.index
            assert stored.x_m == pytest.approx(expected.x_m, abs=1e-15)
            assert stored.y_m == pytest.approx(expected.y_m, abs=1e-15)
