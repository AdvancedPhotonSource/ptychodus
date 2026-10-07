"""The standard single-scan reconstruction, shared by the per-instrument drivers.

Most beamlines reconstruct one scan the same way: read a diffraction series and a position
file, resolve the geometry, condition the patterns, build an initial probe and object, and
hand the result to pty-chi. What differs between instruments is which reader plugins to use,
what the fallback geometry is when a file records none, and how the files are described in
``--help`` -- all of it data. :class:`InstrumentProfile` carries that data and
:func:`run_standard_reconstruction` runs the pipeline, so a driver is a profile and a
``main()`` rather than a copy of the pipeline.

Two instruments vary more than a constant. A reader whose patterns arrive already cropped and
centered sets ``conditions_patterns=False``, which drops the conditioning stage and the
options that drive it. An instrument that wants the MAD and end-trim pattern filters sets
``offers_pattern_filters=True``; those filters are not instrument-specific, so they live here
and the flag only decides who is offered them.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    assemble_dataset,
    compute_dataset_total_counts,
)
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionDataset,
    DiffractionMetadata,
)
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    FilterValuesStep,
)
from ptychodus.api.preprocess.noise import compute_robust_statistics
from ptychodus.api.probe import Probe, ProbeGeometry, ProbeSequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_uniform_object
from ptychodus.api.simulate.probe import (
    generate_average_pattern_probe,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.cli import (
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    positive_int,
)
from ptychodus.cli._reconstruct_common import (
    add_ptychi_options_argument,
    install_signal_handlers,
    invalid_count_threshold,
    load_ptychi_options,
    resolve_crop_region,
    resolve_quantity,
    run_reconstruction,
    save_assembled_diffraction,
)

__all__ = ['InstrumentProfile', 'build_standard_parser', 'run_standard_reconstruction']


@dataclass(frozen=True)
class InstrumentProfile:
    """What distinguishes one instrument's standard reconstruction from another's."""

    logger_name: str
    description: str
    product_name: str
    diffraction_reader: str
    position_reader: str
    diffraction_file_help: str
    position_file_help: str
    fzp_preset_help: str

    # Operating points observed across an instrument's batch scripts. They are a last
    # resort: a value the file records always wins, because the fixture can move between
    # run cycles and the file cannot be stale about itself.
    default_detector_distance_m: float | None = None
    # None means the file must record a pixel size and it is an error when it does not.
    default_detector_pixel_size_m: float | None = None

    default_fzp_preset: str = ''
    default_fzp_defocus_m: float = 0.0
    default_bad_pixels_reader: str = 'NPY_Bad_Pixels'

    # False drops the crop, beam-center, value-filter and bad-pixel options along with the
    # stages that read them, for a reader that hands over patterns already prepared.
    conditions_patterns: bool = True
    # True adds --detector-pixel-size-m, for a layout that records no geometry of its own.
    offers_detector_pixel_size: bool = False
    # True adds --probe-photon-count, for a layout whose recorded flux is worth overriding.
    offers_probe_photon_count: bool = False
    # True adds the MAD and end-trim pattern filters. They are instrument-agnostic; this
    # only decides which instruments offer them.
    offers_pattern_filters: bool = False


def build_standard_parser(profile: InstrumentProfile) -> argparse.ArgumentParser:
    """Build the argument parser `profile` describes."""
    parser = argparse.ArgumentParser(
        description=profile.description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-file',
        required=True,
        type=Path,
        help=profile.diffraction_file_help,
    )
    parser.add_argument(
        '--position-file',
        required=True,
        type=Path,
        help=profile.position_file_help,
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

    if profile.conditions_patterns:
        parser.add_argument(
            '--crop-extent-px',
            type=positive_int,
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
        '--photon-energy-eV',
        type=float,
        default=None,
        help='Photon energy in electron volts. Overrides the file.',
    )

    if profile.offers_detector_pixel_size:
        parser.add_argument(
            '--detector-pixel-size-m',
            type=float,
            default=None,
            help='Detector pixel pitch, both axes. Overrides the file.',
        )

    if profile.conditions_patterns:
        parser.add_argument(
            '--min-total-counts',
            type=int,
            default=None,
            help='Drop patterns whose total counts fall below this, with their positions.',
        )

        if profile.offers_pattern_filters:
            _add_pattern_filter_arguments(parser)

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
            default=profile.default_bad_pixels_reader,
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
        default=profile.default_fzp_preset,
        help=profile.fzp_preset_help,
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=profile.default_fzp_defocus_m,
        help='Defocus from the zone-plate focal plane. Ignored without --fzp-preset.',
    )
    parser.add_argument(
        '--num-probe-modes',
        type=positive_int,
        default=1,
        help='Incoherent probe modes.',
    )
    parser.add_argument(
        '--num-opr-modes',
        type=positive_int,
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
        type=positive_int,
        default=100,
        help='Epochs between reconstructor sync points and progress logs.',
    )

    if profile.offers_probe_photon_count:
        parser.add_argument(
            '--probe-photon-count',
            type=float,
            default=None,
            help='Override the per-snapshot probe photon count.',
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
    add_log_level_argument(parser)
    return parser


def _add_pattern_filter_arguments(parser: argparse.ArgumentParser) -> None:
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


def _mad_bounds(values: numpy.ndarray, k: float) -> tuple[float, float]:
    """Symmetric MAD interval about the median, excluding non-positive values.

    Reproduces the beamline script's ``_mad_mask(x, k, require_positive=True)``, zero-MAD
    degenerate case included: ``get_bounds`` then returns an interval open on the upper
    side, so a constant trace keeps every positive point instead of rejecting all of them.
    """
    interval = compute_robust_statistics(values).get_bounds(k, require_positive=True)
    return interval.lower, interval.upper


def _select_patterns(
    logger: logging.Logger,
    assembled_data: AssembledDiffractionData,
    keep: numpy.ndarray,
    reason: str,
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
    logger: logging.Logger,
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

    if getattr(args, 'dp_mad_k', None) is None:
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
    logger: logging.Logger,
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
    return _select_patterns(logger, assembled_data, keep, 'The --i0-mad-k filter')


def _trim_ends(
    logger: logging.Logger,
    assembled_data: AssembledDiffractionData,
    num_leading: int,
    num_trailing: int,
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

    return _select_patterns(logger, assembled_data, keep, 'The end trim')


def _resolve_detector_pixel_size(
    logger: logging.Logger,
    args: argparse.Namespace,
    metadata: DiffractionMetadata,
    profile: InstrumentProfile,
) -> float:
    """The detector pitch, from the command line, the file, or an instrument default.

    Kept apart from the other two geometry values because the three instrument classes
    disagree about what to do when the file is silent: a layout that records no geometry at
    all offers an override, an instrument with a fixed detector carries a default, and
    everything else treats a missing pitch as an error rather than guessing one.
    """
    pixel_geometry = metadata.detector_pixel_geometry
    from_file = None if pixel_geometry is None else pixel_geometry.width_m

    if profile.offers_detector_pixel_size or profile.default_detector_pixel_size_m is not None:
        return resolve_quantity(
            logger,
            'Detector pixel size (m)',
            getattr(args, 'detector_pixel_size_m', None),
            from_file,
            profile.default_detector_pixel_size_m,
            '--detector-pixel-size-m',
        )

    if from_file is None:
        raise ValueError('The diffraction file records no detector pixel size!')

    logger.info('Detector pixel size (m): %g (from the diffraction file)', from_file)
    return from_file


def _build_initial_probe(
    logger: logging.Logger,
    args: argparse.Namespace,
    registry: PluginRegistry,
    probe_geometry: ProbeGeometry,
    assembled_data: AssembledDiffractionData,
    *,
    photon_wavelength_m: float,
    detector_distance_m: float,
) -> Probe:
    """The probe the reconstruction starts from: a warm start, a zone plate, or the data."""
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

        return Probe(array=array, pixel_geometry=probe_geometry.get_pixel_geometry())

    if args.fzp_preset:
        logger.info('Simulating the initial probe from zone plate preset "%s"', args.fzp_preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)
        return generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            photon_wavelength_m=photon_wavelength_m,
            defocus_distance_m=args.fzp_defocus_m,
        )

    # No zone-plate preset applies, so the cold start comes from the data:
    # back-propagating the mean pattern estimates the probe without asserting an
    # optic at all, which is what lets one script shape serve zone plates, KB
    # mirrors and pinholes alike.
    logger.info('Estimating the initial probe by back-propagating the mean pattern')
    return generate_average_pattern_probe(
        probe_geometry,
        assembled_data,
        photon_wavelength_m=photon_wavelength_m,
        detector_distance_m=detector_distance_m,
    )


def run_standard_reconstruction(profile: InstrumentProfile) -> ExitCode:
    """Reconstruct one scan of `profile`'s instrument, end to end.

    Parses the command line, assembles the dataset, builds an initial product and runs
    pty-chi to convergence, writing the standard layout into ``--output-directory``.

    To reconstruct with something other than stock LSQML, pass a pty-chi options JSON with
    ``--ptychi-options-file``; any ptychodus run writes one to ``ptychi_options.json``, and
    its algorithm stamp selects the reconstructor.
    """
    logger = logging.getLogger(profile.logger_name)
    args = build_standard_parser(profile).parse_args()

    configure_logging(args)

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored. Nothing earlier than this is interruptible.
    cancellation = install_signal_handlers()

    registry = PluginRegistry.load_plugins()
    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(
        profile.diffraction_reader
    )
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(
        profile.position_reader
    )

    logger.info(
        'Reading diffraction from %s as %s', args.diffraction_file, profile.diffraction_reader
    )
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

    detector_distance_m = resolve_quantity(
        logger,
        'Detector distance (m)',
        args.detector_distance_m,
        metadata.detector_distance_m,
        profile.default_detector_distance_m,
        '--detector-distance-m',
    )
    photon_energy_eV = resolve_quantity(  # noqa: N806
        logger,
        'Photon energy (eV)',
        args.photon_energy_eV,
        metadata.photon_energy_eV,
        None,
        '--photon-energy-eV',
    )
    detector_pixel_size_m = _resolve_detector_pixel_size(logger, args, metadata, profile)
    raw_pixel_geometry = PixelGeometry(
        width_m=detector_pixel_size_m, height_m=detector_pixel_size_m
    )

    bad_pixels: BadPixels | None = None
    pipeline: DiffractionPrepPipeline | None = None
    read_region: CropRegion | None = None
    lower_counts: int | None = None
    upper_counts: int | None = None

    if profile.conditions_patterns:
        max_valid_count = (
            invalid_count_threshold(metadata.pattern_dtype)
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

        if args.bad_pixels_file is not None:
            logger.info('Reading bad pixels from %s', args.bad_pixels_file)
            bad_pixels_reader = registry.bad_pixels_file_readers.get_strategy_by_name(
                args.bad_pixels_file_type
            )
            bad_pixels = bad_pixels_reader.read(args.bad_pixels_file)

        read_region = resolve_crop_region(
            logger,
            raw_dataset,
            crop_extent_px=args.crop_extent_px,
            beam_center_x_px=args.beam_center_x_px,
            beam_center_y_px=args.beam_center_y_px,
            bad_pixels=bad_pixels,
            value_filter=value_filter,
        )
        lower_counts, upper_counts = _total_counts_bounds(
            logger, args, raw_dataset, pipeline, bad_pixels, read_region
        )

    logger.info('Assembling diffraction patterns')

    if profile.conditions_patterns:
        assembled_data = assemble_dataset(
            raw_dataset,
            pipeline,
            bad_pixels=bad_pixels,
            raw_pixel_geometry=raw_pixel_geometry,
            read_region=read_region,
            total_counts_lower_bound=lower_counts,
            total_counts_upper_bound=upper_counts,
        )
    else:
        assembled_data = assemble_dataset(raw_dataset, raw_pixel_geometry=raw_pixel_geometry)

    num_patterns = assembled_data.get_num_patterns()

    if num_patterns != num_raw_patterns:
        logger.warning(
            'Assembled %d of %d patterns; %d were filtered or failed to load.',
            num_patterns,
            num_raw_patterns,
            num_raw_patterns - num_patterns,
        )

    logger.info(
        'Reading probe positions from %s as %s', args.position_file, profile.position_reader
    )
    positions = position_reader.read(args.position_file)
    logger.info('Read %d probe positions', len(positions))

    if profile.offers_pattern_filters:
        # I0 lives on the positions, so its filter has to wait until they are read. The end
        # trim runs last so its counts are of patterns that survived both MAD filters.
        if args.i0_mad_k is not None:
            assembled_data = _reject_i0_outliers(logger, assembled_data, positions, args.i0_mad_k)

        if args.drop_leading_frames > 0 or args.drop_trailing_frames > 0:
            assembled_data = _trim_ends(
                logger, assembled_data, args.drop_leading_frames, args.drop_trailing_frames
            )

        num_patterns = assembled_data.get_num_patterns()

    photon_wavelength_m = energy_eV_to_wavelength_m(photon_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=photon_wavelength_m,
        distance_m=detector_distance_m,
    )
    logger.info('Probe geometry: %s', probe_geometry)

    rng = numpy.random.default_rng(args.seed)
    probe = _build_initial_probe(
        logger,
        args,
        registry,
        probe_geometry,
        assembled_data,
        photon_wavelength_m=photon_wavelength_m,
        detector_distance_m=detector_distance_m,
    )

    if args.num_probe_modes > 1:
        probe = generate_incoherent_probe_modes(probe, args.num_probe_modes)

    probe_sequence: ProbeSequence = generate_coherent_probe_modes(
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
    object_ = generate_uniform_object(object_geometry)

    probe_photon_count_override = getattr(args, 'probe_photon_count', None)
    probe_photon_count = (
        float(assembled_data.get_probe_photon_count())
        if probe_photon_count_override is None
        else float(probe_photon_count_override)
    )

    product = Product(
        metadata=ProductMetadata(
            name=profile.product_name,
            comments=f'Reconstructed from {args.diffraction_file.name}',
            detector_distance_m=detector_distance_m,
            photon_energy_eV=photon_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=float(metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=(
                0.0 if args.tomography_angle_deg is None else float(args.tomography_angle_deg)
            ),
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

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
        return ExitCode.SUCCESS

    save_assembled_diffraction(
        logger, output_directory, assembled_data, skip=args.no_save_diffraction
    )

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)
    num_paired = reconstruct_input.diffraction_patterns.shape[0]

    if num_paired < num_patterns:
        # Ordinary on a scan that was stopped mid-raster: the detector keeps the line it
        # was on, the position record truncates at its last completed line, and the
        # patterns past that end have no position to pair with.
        logger.warning(
            'Paired %d of %d assembled patterns; %d had no probe position, which is'
            ' expected when the detector recorded more scan lines than the position file.',
            num_paired,
            num_patterns,
            num_patterns - num_paired,
        )

    run_reconstruction(
        logger,
        reconstruct_input,
        options,
        output_directory,
        num_sync_epochs=args.num_sync_epochs,
        cancellation=cancellation,
    )
    return ExitCode.CANCELLED if cancellation.is_cancelled else ExitCode.SUCCESS
