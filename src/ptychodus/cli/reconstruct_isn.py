#!/usr/bin/env python
"""Reconstruct one APS 19-ID-E In-situ Nanoprobe ptychography dataset through the ptychodus api.

The In-situ Nanoprobe focuses with KB mirrors. Its fly scans expose a single trajectory
positioner; its per-trigger X/Y readback is averaged downstream into a companion
``Processed/SOCKETSERVER/Scan_NNNN_position.h5`` file alongside the
``Raw/Scan_NNNN/PTYCHO/`` diffraction series, read by the ``APS_ISN`` position reader.

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
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy

from ptychodus.api.assemble import assemble_dataset, summarize_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, BeamCenter, CropRegion
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    FilterValuesStep,
    estimate_beam_center,
)
from ptychodus.api.probe import Probe, ProbeGeometry
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
    add_ptychi_options_argument,
    install_signal_handlers,
    load_ptychi_options,
    run_reconstruction,
    save_assembled_diffraction,
)

logger = logging.getLogger('reconstruct_isn')

DIFFRACTION_READER = 'APS_ISN'
POSITION_READER = 'APS_ISN'

# Operating points observed across this instrument's batch scripts. They are a last
# resort: a value the file records always wins, because the fixture can move between
# run cycles and the file cannot be stale about itself. The distance below is never
# recorded in the file and can drift between run cycles -- it is the current-cycle
# value observed in the beamline's own preprocessing scripts, not a fixed constant.
DEFAULT_DETECTOR_DISTANCE_M = 6.16
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


def main() -> ExitCode:
    parser = argparse.ArgumentParser(
        description=(
            'APS 19-ID-E In-situ Nanoprobe ptychography reconstruction via the ptychodus api.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-file',
        required=True,
        type=Path,
        help=(
            'Any member of the PTYCHO/scan_NNNN_FFFFF.h5 (or older 19ide_NNNN_NNN.h5) '
            'series; the rest are globbed.'
        ),
    )
    parser.add_argument(
        '--position-file',
        required=True,
        type=Path,
        help='HDF5 file, normally Processed/SOCKETSERVER/Scan_NNNN_position.h5.',
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
            'FresnelZonePlate plugin preset for the model probe. This instrument focuses with KB '
            'mirrors rather than a zone plate, so a zone-plate probe would be the wrong model '
            'entirely and this is empty by default. Ignored with --probe-file.'
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

    logger.info('Assembling diffraction patterns')
    assembled_data = assemble_dataset(
        raw_dataset,
        pipeline,
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        read_region=read_region,
        total_counts_lower_bound=args.min_total_counts,
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
            name='isn-reconstruct',
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
        return ExitCode.SUCCESS

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
    return ExitCode.CANCELLED if cancellation.is_cancelled else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
