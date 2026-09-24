#!/usr/bin/env python
"""Reconstruct one APS 2-ID-D Bionanoprobe ptychography dataset through the ptychodus api.

The Bionanoprobe writes one Eiger HDF5 per scan line beside an EPICS MDA file. Its positions
are in micrometers, where the 2-ID-E microprobe uses millimeters, and its frames carry enough
invalid-pixel markers that the beam center cannot be estimated without cutting them first.

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

from ptychi.api import LSQMLOptions

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, BeamCenter, CropRegion, DiffractionDataset
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import StandardFileLayout, save_diffraction_data, save_product
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    DiffractionPrepStepUnion,
    FilterValuesStep,
    HorizontalFlipStep,
    TransposeStep,
    VerticalFlipStep,
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
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    dump_task_options,
    reconstruct_with_ptychi,
)

logger = logging.getLogger('reconstruct_bnp')

DIFFRACTION_READER = 'APS_BNP'
POSITION_READER = 'APS_BNP'

# Operating points observed across this instrument's batch scripts. They are a last
# resort: a value the file records always wins, because the fixture can move between
# run cycles and the file cannot be stale about itself.
DEFAULT_DETECTOR_DISTANCE_M = 2.06
DEFAULT_FZP_PRESET = ''

# Where the beam center comes from when neither the command line nor the file supplies
# one. 'estimate' back-propagates nothing -- it centroids the mean of the first array --
# and 'midpoint' takes the detector center, which is right only for a layout that was
# already cropped about the beam by whatever wrote it.
BEAM_CENTER_FALLBACK = 'estimate'


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


# Reading one array uncropped just to locate the beam is cheap when an array is one
# scan line, and ruinous when it is the whole scan: a LamNI acquisition is a single
# 12201 x 1030 x 1614 array, 81 GiB in its native uint32. DiffractionArray exposes no way
# to read a few frames -- get_patterns() crops in space, not in frame -- so the estimate
# is offered only when the first array fits in this budget. Sized to admit a normal
# single-array scan (a 1089-frame ISN scan is about 4.8 GiB) and refuse LamNI.
_ESTIMATE_BUDGET_BYTES = 8 * 1024**3


def _mean_pattern(dataset: DiffractionDataset, upper_bound: int | None) -> numpy.ndarray:
    """Mean over the first array's patterns, for locating the direct beam.

    One array is enough to find the beam and costs one file read; averaging the whole
    scan would mean reading every uncropped frame just to decide where to crop. The same
    invalid-pixel cut the pipeline applies is applied here, so the center is estimated
    from the frame that will actually be reconstructed.
    """
    metadata = dataset.get_metadata()
    extent = metadata.detector_extent
    # Measured in the stored dtype, which is what the read actually allocates.
    wanted_bytes = (
        metadata.num_patterns_per_array[0]
        * extent.width_px
        * extent.height_px
        * metadata.pattern_dtype.itemsize
    )

    if wanted_bytes > _ESTIMATE_BUDGET_BYTES:
        raise ValueError(
            f'Estimating the beam center would read {wanted_bytes / 1024**3:.1f} GiB: this '
            f"dataset's first array holds {metadata.num_patterns_per_array[0]} uncropped "
            f'{extent.width_px}x{extent.height_px} frames. Pass --beam-center-x-px and '
            '--beam-center-y-px instead.'
        )

    patterns = dataset[0].get_patterns()

    # A layout that stores one frame per file hands back a bare 2-D pattern rather than
    # a length-1 stack; averaging over axis 0 would collapse it to a single row.
    if patterns.ndim == 2:
        patterns = patterns[numpy.newaxis]

    # In place, and in the stored dtype: the array came fresh off the reader, so nothing
    # else holds it, and widening the whole stack to float64 first would double the peak
    # for no gain. The accumulator below is float64 regardless.
    if upper_bound is not None:
        patterns[patterns >= upper_bound] = 0

    return patterns.mean(axis=0, dtype=numpy.float64)


def _prep_pipeline(
    args: argparse.Namespace, upper_bound: int | None
) -> DiffractionPrepPipeline | None:
    steps: list[DiffractionPrepStepUnion] = []

    # Invalid-pixel markers and negatives first: every later step, and the assembled
    # photon counts, would otherwise carry them.
    if upper_bound is not None:
        steps.append(FilterValuesStep(lower_bound=0, upper_bound=upper_bound))

    if args.flip_up_down:
        steps.append(VerticalFlipStep())
    if args.flip_left_right:
        steps.append(HorizontalFlipStep())
    if args.transpose:
        steps.append(TransposeStep())

    return DiffractionPrepPipeline(steps=tuple(steps)) if steps else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description='APS 2-ID-D Bionanoprobe ptychography reconstruction via the ptychodus api.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-file',
        required=True,
        type=Path,
        help='Any member of the bnp_flyNNNN_NNNNNN.h5 series; the rest are globbed.',
    )
    parser.add_argument(
        '--position-file',
        required=True,
        type=Path,
        help='EPICS MDA file, normally mda/bnp_flyNNNN.mda.',
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
        '--detector-pixel-size-m',
        type=float,
        default=None,
        help='Detector pixel pitch, both axes. Overrides the file.',
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
    parser.add_argument('--flip-up-down', action='store_true', help='Flip patterns vertically.')
    parser.add_argument(
        '--flip-left-right', action='store_true', help='Flip patterns horizontally.'
    )
    parser.add_argument('--transpose', action='store_true', help='Transpose patterns.')
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
    detector_pixel_size_m = _resolve(
        'Detector pixel size (m)',
        args.detector_pixel_size_m,
        None
        if metadata.detector_pixel_geometry is None
        else metadata.detector_pixel_geometry.width_m,
        None,
        '--detector-pixel-size-m',
    )
    raw_pixel_geometry = PixelGeometry(
        width_m=detector_pixel_size_m, height_m=detector_pixel_size_m
    )

    max_valid_count = (
        _invalid_count_threshold(metadata.pattern_dtype)
        if args.max_valid_count is None
        else args.max_valid_count
    )

    if max_valid_count is not None:
        logger.info('Zeroing pixels at or above %d as invalid', max_valid_count)

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
        elif BEAM_CENTER_FALLBACK == 'midpoint':
            beam_center = BeamCenter(
                x_px=metadata.detector_extent.width_px // 2,
                y_px=metadata.detector_extent.height_px // 2,
            )
            center_source = 'the detector midpoint, which this layout is already cropped about'
        else:
            beam_center = estimate_beam_center(_mean_pattern(raw_dataset, max_valid_count))
            center_source = 'an estimate from the first array'

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

    bad_pixels: BadPixels | None = None

    if args.bad_pixels_file is not None:
        logger.info('Reading bad pixels from %s', args.bad_pixels_file)
        bad_pixels_reader = registry.bad_pixels_file_readers.get_strategy_by_name(
            args.bad_pixels_file_type
        )
        bad_pixels = bad_pixels_reader.read(args.bad_pixels_file)

    logger.info('Assembling diffraction patterns')
    assembled_data = assemble_dataset(
        raw_dataset,
        _prep_pipeline(args, max_valid_count),
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        read_region=read_region,
        total_counts_lower_bound=args.min_total_counts,
    )
    num_patterns = assembled_data.get_patterns().shape[0]

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

    probe_photon_count = (
        float(assembled_data.get_probe_photon_count())
        if args.probe_photon_count is None
        else float(args.probe_photon_count)
    )

    product = Product(
        metadata=ProductMetadata(
            name='bnp-reconstruct',
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

    # USER: customize pty-chi options here (edit fields on `options` before the loop).
    options = LSQMLOptions()

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

    output_directory.mkdir(parents=True, exist_ok=True)

    # Ahead of the reconstruction, so an interrupted run still leaves the directory
    # usable: the assembled patterns are the one artifact that cannot be rebuilt without
    # the raw beamline files.
    if args.no_save_diffraction:
        logger.info('Skipping %s as requested', diffraction_file.name)
    else:
        logger.info('Writing %s', diffraction_file)
        save_diffraction_data(diffraction_file, assembled_data)

    task_options = align_task_options_with_product(options, product)
    task_options.check()

    # The aligned options are the ones that ran: they carry the object pixel size, the
    # wavelength and the slice spacings the product supplied, which the unaligned object
    # does not.
    logger.info('Writing %s', options_file)
    options_file.write_text(dump_task_options(task_options))

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)
    logger.info(
        'Starting reconstruction: %d patterns, %d epochs total, sync every %d',
        reconstruct_input.diffraction_patterns.shape[0],
        num_epochs,
        args.num_sync_epochs,
    )

    final_output = None

    for output in reconstruct_with_ptychi(
        reconstruct_input, task_options, num_sync_epochs=args.num_sync_epochs
    ):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Epoch %d/%d: loss=%.6g', output.progress, num_epochs, last_loss)

        checkpoint_file = StandardFileLayout.PRODUCT.checkpoint_path(
            output_directory, output.progress
        )
        save_product(checkpoint_file, output.product)
        final_output = output

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    save_product(product_file, final_output.product)
    logger.info('Saved reconstructed product to %s', product_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
