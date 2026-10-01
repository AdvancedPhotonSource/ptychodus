#!/usr/bin/env python
"""Reconstruct one APS 4-ID-B,G,H POLAR ptychography dataset through the ptychodus api.

The POLAR master advertises both halves of a scan as external links under
``/entry/externals``: the Eiger frames at ``eiger/scan_NNNNNN.h5`` and, for fly scans, the
softGlueZynq position stream at ``pos_stream/scan_NNNNNN.h5``. Both readers therefore take
the *same* master path, and ``--position-file`` is normally a repeat of
``--diffraction-file``.

How much geometry the file carries depends on when it was written. The newest layout
records the detector distance, the beam center and the mono energy; older ones record only
the energy, so the distance falls back to the built-in default below and the beam center
must be given or estimated. No layout records the detector pixel pitch, which the reader
supplies as the Eiger's 75 um.

Fly scans index positions by the raw softGlueZynq trigger counter, which starts at 0, while
the diffraction reader emits 1-based frame indexes -- so frame ``k`` pairs with trigger
``k + 1`` and trigger 0 is left over. Step scans index both sides off the same Eiger unique
id and pair 1:1. Neither offset is applied by hand: ``prepare_reconstruct_input`` pairs on
the index, which is what makes both conventions work through one script.

Scope and limitations
---------------------

- One scan, LSQML only. Edit the options block near the end to change the algorithm.
- Output is a ptychodus product HDF5. Feeding it back as ``--probe-file`` on the next
  scan is the intended warm start.
- No GPU selection or thread-count side effects. Choose a device with
  ``CUDA_VISIBLE_DEVICES`` in the environment.
- Positions carry their file-provided index, so patterns and positions pair through
  :func:`prepare_reconstruct_input` rather than by array order.
- Geometry resolves in one order -- command line, then the file, then a built-in
  fallback -- and the source of each value is logged.
- The beamline's ``4idd_data_preprocessing_flyscan_v2.py`` normalizes patterns by I0
  (``dp / i0 * 1e5``); this script does not. Ptychodus carries the per-position I0 as
  ``probe_photon_count``, which feeds the illumination map rather than the patterns.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    assemble_dataset,
    compute_dataset_total_counts,
    summarize_dataset,
)
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, BeamCenter, CropRegion, DiffractionDataset
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    FilterValuesStep,
    estimate_beam_center,
)
from ptychodus.api.preprocess.noise import compute_robust_statistics
from ptychodus.api.probe import Probe, ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.simulate.probe import (
    generate_average_pattern_probe,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.cli import DirectoryType
from ptychodus.cli._reconstruct_common import (
    EXIT_CANCELLED,
    add_ptychi_options_argument,
    install_signal_handlers,
    load_ptychi_options,
    run_reconstruction,
    save_assembled_diffraction,
)

logger = logging.getLogger('reconstruct_polar')

DIFFRACTION_READER = 'APS_Polar'
POSITION_READER = 'APS_Polar'

# Operating points observed across this instrument's batch scripts. They are a last
# resort: a value the file records always wins, because the fixture can move between
# run cycles and the file cannot be stale about itself.
DEFAULT_DETECTOR_DISTANCE_M = 1.91
# POLAR focuses with KB mirrors, so no zone-plate preset applies and the cold-start probe
# is estimated from the data instead.
DEFAULT_FZP_PRESET = ''


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')
    return value


def _resolve(
    quantity: str,
    override: float | None,
    from_file: float | None,
    fallback: float | None,
    flag: str,
) -> float:
    """Pick a geometry value from the command line, the file, or a fallback, and say which.

    Reporting the source is the point: a run that silently substituted a built-in
    default for a value the instrument actually recorded would be indistinguishable
    from one that read it.
    """
    if override is not None:
        logger.info('%s: %g (from %s)', quantity, override, flag)
        return override

    if from_file is not None:
        logger.info('%s: %g (from the diffraction file)', quantity, from_file)
        return from_file

    if fallback is not None:
        logger.warning(
            '%s: %g (built-in default for this instrument; the file recorded none). '
            'Pass %s if this scan differs.',
            quantity,
            fallback,
            flag,
        )
        return fallback

    raise ValueError(f'{quantity} is not in the file and has no default; pass {flag}.')


def _invalid_count_threshold(dtype: numpy.dtype) -> int | None:
    """Count at and above which a pixel is invalid rather than merely bright.

    Dectris detectors flag dead, masked and saturated pixels with the maximum value the
    pattern dtype can hold -- 4294967295 on a uint32 Eiger. Left in place those markers
    dominate every statistic computed from the frame: the beam-center estimator rejects
    the real beam as noise and silently returns the detector midpoint instead.
    """
    if numpy.issubdtype(dtype, numpy.integer):
        return int(numpy.iinfo(dtype).max)

    return None


def _mad_bounds(values: numpy.ndarray, k: float) -> tuple[float, float]:
    """Symmetric MAD interval about the median, excluding non-positive values.

    Reproduces the beamline script's ``_mad_mask(x, k, require_positive=True)``, zero-MAD
    degenerate case included: ``get_bounds`` then returns an interval open on the upper
    side, so a constant trace keeps every positive point instead of rejecting all of them.
    """
    interval = compute_robust_statistics(values).get_bounds(k, require_positive=True)
    return interval.lower, interval.upper


def _select_patterns(
    assembled_data: AssembledDiffractionData, keep: numpy.ndarray, reason: str
) -> AssembledDiffractionData:
    """Return the patterns `keep` selects, as a new AssembledDiffractionData.

    Patterns are dropped rather than positions. Dropping a position would not remove its
    pattern: `prepare_reconstruct_input` interpolates a pattern index that falls inside the
    position range but has no exact match, so the frame would come back with a made-up
    coordinate instead of being discarded.
    """
    num_dropped = int((~keep).sum())

    if num_dropped == 0:
        logger.info('%s dropped no patterns.', reason)
        return assembled_data

    if not keep.any():
        raise ValueError(f'{reason} dropped every pattern. Loosen or remove the option.')

    logger.warning(
        '%s dropped %d of %d patterns (kept %d).',
        reason,
        num_dropped,
        keep.size,
        int(keep.sum()),
    )

    photon_counts = (
        assembled_data.get_probe_photon_counts()[keep]
        if assembled_data.has_measured_probe_photon_counts()
        else None
    )

    return AssembledDiffractionData(
        indexes=assembled_data.get_indexes()[keep],
        patterns=assembled_data.get_patterns()[keep],
        pixel_geometry=assembled_data.get_pixel_geometry(),
        bad_pixels=assembled_data.get_bad_pixels(),
        probe_photon_counts=photon_counts,
    )


def _total_counts_bounds(
    args: argparse.Namespace,
    raw_dataset: DiffractionDataset,
    pipeline: DiffractionPrepPipeline | None,
    bad_pixels: BadPixels | None,
    read_region: CropRegion | None,
) -> tuple[int | None, int | None]:
    """Resolve the total-counts window from --min-total-counts and --dp-mad-k.

    The MAD bound needs the distribution before it can pick a threshold, so this makes a
    measuring pass with `compute_dataset_total_counts` -- same crop, same pipeline, same
    processed bad-pixel mask as the assembler, and no assembled buffer. Whichever of the
    two options gives the tighter lower bound wins, so they compose rather than override.
    """
    lower = args.min_total_counts
    upper: int | None = None

    if args.dp_mad_k is None:
        return lower, upper

    logger.info('Measuring per-pattern total counts for the --dp-mad-k window')
    measured = compute_dataset_total_counts(
        raw_dataset, pipeline, bad_pixels=bad_pixels, read_region=read_region
    )
    mad_lower, mad_upper = _mad_bounds(measured.total_counts.astype(float), args.dp_mad_k)

    # The filter compares integer counts inclusively, so widen each bound outward to the
    # nearest integer; rounding inward would reject patterns the float interval admits.
    if math.isfinite(mad_lower):
        mad_lower_int = int(math.floor(mad_lower))
        lower = mad_lower_int if lower is None else max(lower, mad_lower_int)

    if math.isfinite(mad_upper):
        upper = int(math.ceil(mad_upper))

    logger.info(
        'Total counts (median %.4g, MAD %.4g over %d patterns): keeping [%s, %s]',
        float(numpy.median(measured.total_counts)),
        float(numpy.median(numpy.abs(measured.total_counts - numpy.median(measured.total_counts)))),
        measured.total_counts.size,
        lower,
        upper,
    )
    return lower, upper


def _reject_i0_outliers(
    assembled_data: AssembledDiffractionData,
    positions: ProbePositionSequence,
    mad_k: float,
) -> AssembledDiffractionData:
    """Drop patterns whose position-side I0 is a MAD outlier.

    I0 is recorded per position, not per pattern, so the rejection is expressed as a set of
    scan indexes and then applied to the patterns. Only fly scans carry it: the softGlueZynq
    stream reduces the I0 counter over each trigger group, while step-scan motor readbacks
    have no counter to reduce.
    """
    counts = positions.get_probe_photon_counts()

    if counts is None:
        logger.warning(
            'Ignoring --i0-mad-k: these positions carry no I0. Only fly scans record it.'
        )
        return assembled_data

    lower, upper = _mad_bounds(numpy.asarray(counts, dtype=float), mad_k)
    rejected = {
        int(position.index)
        for position, count in zip(positions, counts)
        if not lower <= float(count) <= upper
    }
    logger.info(
        'I0 (%d positions): keeping [%.6g, %.6g]; %d positions rejected',
        len(counts),
        lower,
        upper,
        len(rejected),
    )
    keep = numpy.array(
        [int(index) not in rejected for index in assembled_data.get_indexes()], dtype=bool
    )
    return _select_patterns(assembled_data, keep, 'The --i0-mad-k filter')


def _trim_ends(
    assembled_data: AssembledDiffractionData, num_leading: int, num_trailing: int
) -> AssembledDiffractionData:
    """Drop the first and last patterns of the trajectory.

    The scanner takes a few stabilization frames as the stage comes up to speed, and may
    take a shutter-close frame at the end. This runs after the MAD filters, so the counts
    are of surviving patterns, matching the beamline script's ``_start_drop_mask`` applied
    to the combined keep mask.
    """
    num_patterns = assembled_data.get_num_patterns()

    if num_leading + num_trailing >= num_patterns:
        raise ValueError(
            f'Trimming {num_leading} leading and {num_trailing} trailing patterns would '
            f'consume all {num_patterns} of them.'
        )

    keep = numpy.ones(num_patterns, dtype=bool)

    if num_leading > 0:
        keep[:num_leading] = False

    if num_trailing > 0:
        keep[num_patterns - num_trailing :] = False

    return _select_patterns(assembled_data, keep, 'The end trim')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='APS 4-ID-B,G,H POLAR ptychography reconstruction via the ptychodus api.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-file',
        required=True,
        type=Path,
        help='POLAR master HDF5 file (scan_NNNNNN_master.hdf).',
    )
    parser.add_argument(
        '--position-file',
        required=True,
        type=Path,
        help='POLAR master HDF5 file; normally the same path as --diffraction-file.',
    )
    parser.add_argument(
        '--output-directory',
        required=True,
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory, written in the ptychodus standard layout: '
            'diffraction.h5, ptychi_options.json, per-epoch product.NNNNNN.h5 '
            'checkpoints, and the final product.h5.'
        ),
    )
    parser.add_argument(
        '--no-save-diffraction',
        action='store_true',
        help='Skip diffraction.h5. The assembled patterns can run to several GB.',
    )
    parser.add_argument(
        '--crop-extent-px',
        type=_positive_int,
        default=None,
        help='Square crop side in raw detector pixels, about the beam center. Omit to skip.',
    )
    parser.add_argument(
        '--beam-center-x-px',
        type=int,
        default=None,
        help='Beam center column. File value, then an estimate from the data, otherwise.',
    )
    parser.add_argument(
        '--beam-center-y-px',
        type=int,
        default=None,
        help='Beam center row. File value, then an estimate from the data, otherwise.',
    )
    parser.add_argument(
        '--detector-distance-m',
        type=float,
        default=None,
        help='Sample-to-detector distance. Overrides the file.',
    )
    parser.add_argument(
        '--probe-energy-eV',
        type=float,
        default=None,
        help='Probe energy in electron volts. Overrides the file.',
    )
    parser.add_argument(
        '--min-total-counts',
        type=int,
        default=None,
        help='Drop patterns whose total counts fall below this, with their positions.',
    )
    parser.add_argument(
        '--dp-mad-k',
        type=float,
        default=None,
        help=(
            'Drop patterns whose total counts are more than this many MADs from the median. '
            'The beamline script uses 5.0. Omit to skip.'
        ),
    )
    parser.add_argument(
        '--i0-mad-k',
        type=float,
        default=None,
        help=(
            'Drop patterns whose position-side I0 is more than this many MADs from the median. '
            'The beamline script uses 5.0. Fly scans only. Omit to skip.'
        ),
    )
    parser.add_argument(
        '--drop-leading-frames',
        type=int,
        default=0,
        help='Drop this many patterns from the start of the trajectory, after the MAD filters.',
    )
    parser.add_argument(
        '--drop-trailing-frames',
        type=int,
        default=0,
        help='Drop this many patterns from the end of the trajectory, after the MAD filters.',
    )
    parser.add_argument(
        '--max-valid-count',
        type=int,
        default=None,
        help=(
            'Zero pixels at or above this count. Defaults to the pattern dtype maximum, '
            'which is how Dectris detectors mark invalid pixels.'
        ),
    )
    parser.add_argument(
        '--bad-pixels-file',
        type=Path,
        default=None,
        help='Bad-pixel mask covering the full, uncropped detector.',
    )
    parser.add_argument(
        '--bad-pixels-file-type',
        default='NPY_Bad_Pixels',
        help='Name of the bad-pixel file reader plugin.',
    )
    parser.add_argument(
        '--probe-file',
        type=Path,
        default=None,
        help=(
            'Initial probe, normally a product HDF5 from a previous scan. '
            'A model probe is built when omitted.'
        ),
    )
    parser.add_argument(
        '--probe-file-type',
        default='HDF5',
        help='Name of the probe file reader plugin; HDF5 reads a ptychodus product.',
    )
    parser.add_argument(
        '--fzp-preset',
        default=DEFAULT_FZP_PRESET,
        help=(
            'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is verified '
            'for this instrument, so this is empty by default and the probe is estimated from the '
            'data instead. Ignored with --probe-file.'
        ),
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=0.0,
        help='Defocus from the zone-plate focal plane. Ignored without --fzp-preset.',
    )
    parser.add_argument(
        '--num-probe-modes',
        type=_positive_int,
        default=1,
        help='Incoherent probe modes.',
    )
    parser.add_argument(
        '--num-opr-modes',
        type=_positive_int,
        default=1,
        help='Coherent (OPR) probe modes; 1 disables OPR.',
    )
    parser.add_argument(
        '--object-padding-px',
        type=int,
        default=64,
        help='Object-canvas margin added to each side of the scan bounding box.',
    )
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for the initial guesses.')
    parser.add_argument(
        '--num-sync-epochs',
        type=_positive_int,
        default=100,
        help='Epochs between reconstructor sync points and progress logs.',
    )
    parser.add_argument(
        '--tomography-angle-deg',
        type=float,
        default=None,
        help='Override the tomography angle in degrees.',
    )
    add_ptychi_options_argument(parser)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build everything and report the resolved configuration without reconstructing.',
    )
    parser.add_argument(
        '--log-level',
        default=logging.INFO,
        type=int,
        help='Python logging level.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored. Nothing earlier than this is interruptible.
    cancellation = install_signal_handlers()

    registry = PluginRegistry.load_plugins()
    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(DIFFRACTION_READER)
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(POSITION_READER)

    logger.info('Reading diffraction from %s as %s', args.diffraction_file, DIFFRACTION_READER)
    raw_dataset = diffraction_reader.read(args.diffraction_file)
    metadata = raw_dataset.get_metadata()

    num_raw_patterns = sum(metadata.num_patterns_per_array)
    logger.info(
        'Series: %d array(s), %d pattern(s), detector %dx%d',
        len(metadata.num_patterns_per_array),
        num_raw_patterns,
        metadata.detector_extent.width_px,
        metadata.detector_extent.height_px,
    )

    detector_distance_m = _resolve(
        'Detector distance (m)',
        args.detector_distance_m,
        metadata.detector_distance_m,
        DEFAULT_DETECTOR_DISTANCE_M,
        '--detector-distance-m',
    )
    probe_energy_eV = _resolve(  # noqa: N806
        'Probe energy (eV)',
        args.probe_energy_eV,
        metadata.probe_energy_eV,
        None,
        '--probe-energy-eV',
    )
    if metadata.detector_pixel_geometry is None:
        raise ValueError('The diffraction file records no detector pixel size!')

    detector_pixel_size_m = metadata.detector_pixel_geometry.width_m
    logger.info('Detector pixel size (m): %g (from the diffraction file)', detector_pixel_size_m)
    raw_pixel_geometry = PixelGeometry(
        width_m=detector_pixel_size_m, height_m=detector_pixel_size_m
    )

    max_valid_count = (
        _invalid_count_threshold(metadata.pattern_dtype)
        if args.max_valid_count is None
        else args.max_valid_count
    )

    # One step, two consumers: the assembled patterns below and the mean frame the beam
    # center is estimated from. Sharing the object is what keeps those two cuts identical.
    value_filter: FilterValuesStep | None = None

    if max_valid_count is not None:
        logger.info('Zeroing pixels at or above %d as invalid', max_valid_count)
        value_filter = FilterValuesStep(lower_bound=0, upper_bound=max_valid_count)

    pipeline = None if value_filter is None else DiffractionPrepPipeline(steps=(value_filter,))

    bad_pixels: BadPixels | None = None

    if args.bad_pixels_file is not None:
        logger.info('Reading bad pixels from %s', args.bad_pixels_file)
        bad_pixels_reader = registry.bad_pixels_file_readers.get_strategy_by_name(
            args.bad_pixels_file_type
        )
        bad_pixels = bad_pixels_reader.read(args.bad_pixels_file)

    read_region: CropRegion | None = None

    if args.crop_extent_px is not None:
        # Beam center precedence: command line, then the file, then an estimate from the
        # data. The estimate is logged as such -- a wrong center crops the wrong part of
        # the detector, and nothing downstream would reveal it.
        if args.beam_center_x_px is not None and args.beam_center_y_px is not None:
            beam_center = BeamCenter(x_px=args.beam_center_x_px, y_px=args.beam_center_y_px)
            center_source = '--beam-center-{x,y}-px'
        elif metadata.beam_center is not None:
            beam_center = metadata.beam_center
            center_source = 'the diffraction file'
        else:
            # Reads every pattern in the dataset, so it is reached only when no center was
            # supplied by flag or file. summarize_dataset inpaints the bad pixels that
            # estimate_beam_center requires the caller to have handled.
            logger.info('Summarizing the dataset to estimate the beam center')
            summary = summarize_dataset(raw_dataset, bad_pixels=bad_pixels)
            mean_pattern = (
                summary.mean_pattern
                if value_filter is None
                else value_filter.apply(summary.mean_pattern)
            )
            beam_center = estimate_beam_center(mean_pattern)
            center_source = 'an estimate over the whole dataset'

        logger.info(
            'Beam center: (%d, %d) from %s', beam_center.x_px, beam_center.y_px, center_source
        )

        extent = ImageExtent(width_px=args.crop_extent_px, height_px=args.crop_extent_px)

        if extent != metadata.detector_extent:
            region = CropRegion.from_center_extent(beam_center, extent)

            # from_center_extent does not clip. Silently clamping would quietly reconstruct
            # a different region than asked for, so an overhanging crop is an error.
            if region.clamp_to_detector_extent(metadata.detector_extent) != region:
                raise ValueError(
                    f'A {args.crop_extent_px}px crop about ({beam_center.x_px}, '
                    f'{beam_center.y_px}) runs off the '
                    f'{metadata.detector_extent.width_px}x'
                    f'{metadata.detector_extent.height_px} detector '
                    f'(x={region.x_range} y={region.y_range}). '
                    'Give a smaller --crop-extent-px or an explicit beam center.'
                )

            logger.info('Cropping to x=%s y=%s', region.x_range, region.y_range)
            read_region = region

    lower_counts, upper_counts = _total_counts_bounds(
        args, raw_dataset, pipeline, bad_pixels, read_region
    )

    logger.info('Assembling diffraction patterns')
    assembled_data = assemble_dataset(
        raw_dataset,
        pipeline,
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        read_region=read_region,
        total_counts_lower_bound=lower_counts,
        total_counts_upper_bound=upper_counts,
    )
    num_patterns = assembled_data.get_num_patterns()

    if num_patterns != num_raw_patterns:
        logger.warning(
            'Assembled %d of %d patterns; %d were filtered or failed to load.',
            num_patterns,
            num_raw_patterns,
            num_raw_patterns - num_patterns,
        )

    logger.info('Reading probe positions from %s as %s', args.position_file, POSITION_READER)
    positions = position_reader.read(args.position_file)
    logger.info('Read %d probe positions', len(positions))

    # I0 lives on the positions, so its filter has to wait until they are read. The end
    # trim runs last so its counts are of patterns that survived both MAD filters.
    if args.i0_mad_k is not None:
        assembled_data = _reject_i0_outliers(assembled_data, positions, args.i0_mad_k)

    if args.drop_leading_frames > 0 or args.drop_trailing_frames > 0:
        assembled_data = _trim_ends(
            assembled_data, args.drop_leading_frames, args.drop_trailing_frames
        )

    num_patterns = assembled_data.get_num_patterns()

    probe_wavelength_m = energy_eV_to_wavelength_m(probe_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=probe_wavelength_m,
        distance_m=detector_distance_m,
    )
    logger.info('Probe geometry: %s', probe_geometry)

    rng = numpy.random.default_rng(args.seed)

    if args.probe_file is not None:
        logger.info('Reading the initial probe from %s', args.probe_file)
        probe_reader = registry.probe_file_readers.get_strategy_by_name(args.probe_file_type)
        probe_from_file = probe_reader.read(args.probe_file)
        array = probe_from_file.get_probe_no_opr().get_array()

        # The warm-start probe must sit on the same grid as the patterns it will be
        # refined against; a mismatch means the crop changed since it was written.
        if array.shape[-2:] != (probe_geometry.height_px, probe_geometry.width_px):
            raise ValueError(
                f'Probe file extent {array.shape[-2]}x{array.shape[-1]} does not match the '
                f'patterns {probe_geometry.height_px}x{probe_geometry.width_px}. '
                'Reconstruct with the same --crop-extent-px that produced it.'
            )

        probe = Probe(array=array, pixel_geometry=probe_geometry.get_pixel_geometry())
    elif args.fzp_preset:
        logger.info('Simulating the initial probe from zone plate preset "%s"', args.fzp_preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)
        probe = generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            probe_wavelength_m=probe_wavelength_m,
            defocus_distance_m=args.fzp_defocus_m,
        )
    else:
        # No zone-plate preset applies, so the cold start comes from the data:
        # back-propagating the mean pattern estimates the probe without asserting an
        # optic at all, which is what lets one script shape serve zone plates, KB
        # mirrors and pinholes alike.
        logger.info('Estimating the initial probe by back-propagating the mean pattern')
        probe = generate_average_pattern_probe(
            probe_geometry,
            assembled_data,
            probe_wavelength_m=probe_wavelength_m,
            detector_distance_m=detector_distance_m,
        )

    if args.num_probe_modes > 1:
        # Geometric weights: each successive incoherent mode carries half the power of
        # the one before it.
        weights = [0.5**imode for imode in range(args.num_probe_modes)]
        probe = generate_incoherent_probe_modes(rng, probe, weights)

    probe_sequence = generate_coherent_probe_modes(
        rng,
        probe,
        num_cmodes=args.num_opr_modes,
        num_diffraction_patterns=num_patterns,
    )
    logger.info('Probe: %s', probe_sequence.get_array().shape)

    object_geometry = compute_object_geometry(
        positions, probe_geometry, padding_px=args.object_padding_px
    )
    logger.info('Object geometry: %s', object_geometry)

    # Deviations = 0 gives a flat unit-amplitude, zero-phase field.
    object_ = generate_random_object(
        rng,
        object_geometry,
        amplitude_mean=1.0,
        amplitude_deviation=0.0,
        phase_mean=0.0,
        phase_deviation_tr=0.0,
        blur_deviation_px=0.0,
    )

    probe_photon_count = float(assembled_data.get_probe_photon_count())

    product = Product(
        metadata=ProductMetadata(
            name='polar-reconstruct',
            comments=f'Reconstructed from {args.diffraction_file.name}',
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=float(metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=(
                0.0 if args.tomography_angle_deg is None else float(args.tomography_angle_deg)
            ),
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

    # USER: customize pty-chi options here (edit fields on `options` before the loop), or
    # pass a JSON file written by an earlier run with --ptychi-options-file.
    options = load_ptychi_options(args.ptychi_options_file)

    num_epochs = int(options.reconstructor_options.num_epochs)

    output_directory = args.output_directory
    diffraction_file = StandardFileLayout.DIFFRACTION.path(output_directory)
    options_file = StandardFileLayout.PTYCHI_OPTIONS.path(output_directory)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)

    if args.dry_run:
        # Stops ahead of align_task_options_with_product, so a dry run exercises the
        # ptychodus side -- readers, geometry, probe, object -- without requiring a
        # pty-chi whose options match this build.
        logger.info(
            'Dry run: %d patterns, %d epochs, probe %s, object %s.',
            num_patterns,
            num_epochs,
            probe_sequence.get_array().shape,
            object_.get_array().shape,
        )
        logger.info('Would write %s', product_file)
        logger.info(
            'Would write checkpoints like %s',
            StandardFileLayout.PRODUCT.checkpoint_path(output_directory, args.num_sync_epochs),
        )

        if not args.no_save_diffraction:
            logger.info('Would write %s', diffraction_file)

        # The options file records the ALIGNED options, which only exist past the call
        # this dry run stops before, so it is the one artifact a dry run cannot preview.
        logger.info('Would write %s once the options are aligned', options_file)
        logger.info('Nothing written: stopping before the pty-chi option alignment.')
        return 0

    save_assembled_diffraction(
        logger, output_directory, assembled_data, skip=args.no_save_diffraction
    )

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)
    run_reconstruction(
        logger,
        reconstruct_input,
        options,
        output_directory,
        num_sync_epochs=args.num_sync_epochs,
        cancellation=cancellation,
    )
    return EXIT_CANCELLED if cancellation.is_cancelled else 0


if __name__ == '__main__':
    sys.exit(main())
