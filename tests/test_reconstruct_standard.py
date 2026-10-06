"""The shared standard-reconstruction pipeline and the profiles the drivers declare.

The parser tests matter because nine beamline commands now get their entire command-line
surface from one builder: a mistake in a profile field is a silent change to a published
CLI, and nothing else would catch it. The filter tests cover the only arithmetic in the
module.
"""

from __future__ import annotations

import argparse
import logging

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.assemble import AssembledDiffractionData  # noqa: E402
from ptychodus.api.diffraction import (  # noqa: E402
    DiffractionIndexes,
    DiffractionPatternCounts,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence  # noqa: E402
from ptychodus.cli import _reconstruct_standard as rs  # noqa: E402
from ptychodus.cli._reconstruct_standard import InstrumentProfile  # noqa: E402

from diffraction_dataset_fixtures import FailingArray, make_dataset  # noqa: E402

logger = logging.getLogger('test_reconstruct_standard')


def _profiles() -> dict[str, InstrumentProfile]:
    import importlib

    names = (
        '12ide',
        '2ide',
        'atomic',
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


def test_instrument_defaults_are_carried_from_the_profile() -> None:
    profiles = _profiles()

    assert _defaults(profiles['12ide']).detector_distance_m is None
    assert profiles['12ide'].default_detector_distance_m == 10.2
    assert profiles['bnp'].default_detector_distance_m == 2.06
    assert profiles['isn'].default_detector_distance_m == 6.16

    # lamni is the one instrument with a non-zero zone-plate defocus.
    assert _defaults(profiles['lamni']).fzp_defocus_m == 800e-6
    assert _defaults(profiles['12ide']).fzp_defocus_m == 0.0

    # atomic is the one instrument with its own bad-pixel reader and a pixel-size fallback.
    assert _defaults(profiles['atomic']).bad_pixels_file_type == 'APS_Atomic_Bad_Pixels'
    assert _defaults(profiles['12ide']).bad_pixels_file_type == 'NPY_Bad_Pixels'
    assert profiles['atomic'].default_detector_pixel_size_m == 75e-6

    # velociprobe is the one instrument whose two readers differ.
    assert profiles['velociprobe'].diffraction_reader == 'APS_Velociprobe'
    assert profiles['velociprobe'].position_reader == 'APS_Velociprobe_PE'


def test_unconditioned_profile_drops_the_conditioning_options() -> None:
    args = _defaults(_profiles()['foldslice'])

    for absent in (
        'crop_extent_px',
        'beam_center_x_px',
        'beam_center_y_px',
        'min_total_counts',
        'max_valid_count',
        'bad_pixels_file',
        'bad_pixels_file_type',
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
    assert polar.drop_leading_frames == 0
    assert polar.drop_trailing_frames == 0

    for name in ('12ide', '2ide', 'atomic', 'bnp', 'isn', 'lamni', 'velociprobe'):
        args = _defaults(profiles[name])
        for absent in ('dp_mad_k', 'i0_mad_k', 'drop_leading_frames', 'drop_trailing_frames'):
            assert not hasattr(args, absent), f'{name}.{absent}'


def test_logger_names_stay_per_driver() -> None:
    """Log provenance is what makes a beamline log readable; the shared body must not eat it."""
    for name, profile in _profiles().items():
        assert profile.logger_name == f'reconstruct_{name}'


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


def _assembled(
    indexes: DiffractionIndexes, counts: DiffractionPatternCounts | None = None
) -> AssembledDiffractionData:
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
        probe_photon_counts=counts,
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

    assert positions.get_probe_photon_counts() is None
    assert rs._reject_i0_outliers(logger, data, positions, 5.0) is data


def test_total_counts_bounds_passes_min_through_when_mad_is_off() -> None:
    args = argparse.Namespace(min_total_counts=42, dp_mad_k=None)
    assert rs._total_counts_bounds(logger, args, _never_read_dataset(), None, None, None) == (
        42,
        None,
    )


def test_total_counts_bounds_is_inert_when_both_options_are_off() -> None:
    args = argparse.Namespace(min_total_counts=None, dp_mad_k=None)
    assert rs._total_counts_bounds(logger, args, _never_read_dataset(), None, None, None) == (
        None,
        None,
    )
