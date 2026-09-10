#!/usr/bin/env python
"""Reconstruct one APS 31-ID-E LamNI ptychography dataset through the ptychodus api."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy

from ptychi.api.options.lsqml import (
    LSQMLObjectOptions,
    LSQMLOPRModeWeightsOptions,
    LSQMLProbeOptions,
    LSQMLProbePositionOptions,
    LSQMLReconstructorOptions,
)
from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, CropRegion
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import save_product
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeGeometry, ProbeSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.simulate.probe import generate_fresnel_zone_plate_probe
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    reconstruct_with_ptychi,
)

logger = logging.getLogger('reconstruct_lamni')


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description='LamNI ptychography reconstruction via the ptychodus api.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-file',
        required=True,
        type=Path,
        help='Raw LamNI HDF5 file (APS 31-ID-E).',
    )
    parser.add_argument(
        '--position-file',
        required=True,
        type=Path,
        help='LamNI probe-position .dat file (Orchestra or softGlueZynq).',
    )
    parser.add_argument(
        '--output-product',
        required=True,
        type=Path,
        help='Destination path for the reconstructed product HDF5.',
    )
    # The crop extent and the probe extent must agree, and the probe file is the
    # more authoritative of the two, so it supplies the crop rather than being
    # checked against one the user computed by hand.
    crop_source_group = parser.add_mutually_exclusive_group()
    crop_source_group.add_argument(
        '--crop-extent-px',
        type=int,
        default=None,
        help=(
            'Square crop side in raw detector pixels; uses the HDF5 beam center. '
            'Omit to skip. Mutually exclusive with --probe-file.'
        ),
    )
    parser.add_argument(
        '--detector-distance-m',
        type=float,
        default=None,
        help='Override the sample-to-detector distance in meters.',
    )
    parser.add_argument(
        '--probe-energy-eV',
        type=float,
        default=None,
        help='Override the probe energy in electron volts.',
    )
    parser.add_argument(
        '--detector-pixel-size-m',
        type=float,
        default=None,
        help='Override the detector pixel size in meters; applied to both axes.',
    )
    parser.add_argument(
        '--bad-pixels-file',
        type=Path,
        default=None,
        help='Bad-pixel mask covering the full, uncropped detector; the reader mask otherwise.',
    )
    parser.add_argument(
        '--bad-pixels-file-type',
        default='NPY_Bad_Pixels',
        help='Name of the bad-pixel file reader plugin to use.',
    )
    crop_source_group.add_argument(
        '--probe-file',
        type=Path,
        default=None,
        help=(
            'Initial probe; the simulated FZP probe is used when omitted. '
            'Its extent determines the crop. Mutually exclusive with --crop-extent-px.'
        ),
    )
    parser.add_argument(
        '--probe-file-type',
        default='NPY',
        help='Name of the probe file reader plugin to use.',
    )
    parser.add_argument(
        '--fzp-preset',
        default='APS 31-ID-E LamNI',
        help='Name of the FresnelZonePlate plugin preset to use; ignored with --probe-file.',
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=800e-6,
        help=(
            'Defocus distance from the FZP focal plane; default matches the LYNX config. '
            'Ignored with --probe-file.'
        ),
    )
    parser.add_argument(
        '--object-padding-px',
        type=int,
        default=64,
        help='Object-canvas margin, in pixels, added to each side of the scan bounding box.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for the initial object.',
    )
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
        help='Override the per-snapshot probe photon count (assembled-data max otherwise).',
    )
    parser.add_argument(
        '--tomography-angle-deg',
        type=float,
        default=None,
        help='Override the tomography angle in degrees.',
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
    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name('APS_LamNI')
    position_reader = registry.probe_position_file_readers.get_strategy_by_name('APS_LamNI')

    logger.info('Reading raw diffraction from %s', args.diffraction_file)
    raw_dataset = diffraction_reader.read(args.diffraction_file)
    metadata = raw_dataset.get_metadata()

    detector_distance_m = (
        metadata.detector_distance_m
        if args.detector_distance_m is None
        else args.detector_distance_m
    )
    if detector_distance_m is None:
        raise ValueError(
            'Detector distance is missing from the HDF5 metadata; pass --detector-distance-m.'
        )

    probe_energy_eV = (  # noqa: N806
        metadata.probe_energy_eV if args.probe_energy_eV is None else args.probe_energy_eV
    )
    if probe_energy_eV is None:
        raise ValueError('Probe energy is missing from the HDF5 metadata; pass --probe-energy-eV.')

    if args.detector_pixel_size_m is None:
        raw_pixel_geometry = metadata.detector_pixel_geometry
    else:
        raw_pixel_geometry = PixelGeometry(
            width_m=args.detector_pixel_size_m,
            height_m=args.detector_pixel_size_m,
        )
    if raw_pixel_geometry is None:
        raise ValueError(
            'Detector pixel geometry is missing from the HDF5 metadata; '
            'pass --detector-pixel-size-m.'
        )

    # Read the probe before assembly: argparse makes --probe-file and
    # --crop-extent-px mutually exclusive, so a probe file is the crop source.
    probe_from_file: ProbeSequence | None = None
    if args.probe_file is not None:
        logger.info('Reading the initial probe from %s', args.probe_file)
        probe_reader = registry.probe_file_readers.get_strategy_by_name(args.probe_file_type)
        probe_from_file = probe_reader.read(args.probe_file)

    crop_extent: ImageExtent | None = None
    crop_source = ''

    if probe_from_file is not None:
        crop_extent = ImageExtent(
            width_px=probe_from_file.width_px,
            height_px=probe_from_file.height_px,
        )
        crop_source = f'the probe file {args.probe_file}'
    elif args.crop_extent_px is not None:
        crop_extent = ImageExtent(width_px=args.crop_extent_px, height_px=args.crop_extent_px)
        crop_source = '--crop-extent-px'

    # An extent matching the detector needs no crop; re-centering a full-frame probe
    # on the beam center would shift the frame and then fail the bounds check below.
    read_region: CropRegion | None = None
    if crop_extent is not None and crop_extent != metadata.detector_extent:
        if metadata.beam_center is None:
            raise ValueError(
                f'A crop extent came from {crop_source}, but the HDF5 provides no beam center.'
            )

        region = CropRegion.from_center_extent(metadata.beam_center, crop_extent)

        # from_center_extent does not clip, so an oversized extent or an edge-adjacent
        # beam center yields negative slice starts and a silently wrong region. Compare
        # against the clamped form rather than redoing the bounds arithmetic here.
        if region.clamp_to_detector_extent(metadata.detector_extent) != region:
            raise ValueError(
                f'The crop region from {crop_source} does not fit the detector! '
                f'(x={region.x_range} y={region.y_range} '
                f'detector={metadata.detector_extent.width_px}x'
                f'{metadata.detector_extent.height_px} '
                f'beam center=({metadata.beam_center.x_px}, {metadata.beam_center.y_px}))'
            )

        logger.info('Cropping to x=%s y=%s from %s', region.x_range, region.y_range, crop_source)
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
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        read_region=read_region,
    )

    logger.info('Reading probe positions from %s', args.position_file)
    positions = position_reader.read(args.position_file)

    probe_wavelength_m = energy_eV_to_wavelength_m(probe_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=probe_wavelength_m,
        distance_m=detector_distance_m,
    )
    logger.info('Probe geometry: %s', probe_geometry)

    if probe_from_file is None:
        logger.info('Simulating the initial probe from FZP preset "%s"', args.fzp_preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)
        probe = generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            probe_wavelength_m=probe_wavelength_m,
            defocus_distance_m=args.fzp_defocus_m,
        )
        probe_sequence = ProbeSequence.from_probe(probe)
    else:
        derived_pixel_geometry = probe_geometry.get_pixel_geometry()

        # A reader that reports no pixel geometry inherits the one derived from the
        # detector; see FromFileProbeBuilder in model/product/probe/builder.py. When
        # it does report one, it must agree: compute_object_geometry below is handed
        # the derived geometry, so a disagreement would silently put the probe and
        # the object canvas on different grids.
        try:
            probe_pixel_geometry = probe_from_file.get_pixel_geometry()
        except ValueError:
            probe_pixel_geometry = derived_pixel_geometry
        else:
            if not (
                math.isclose(
                    probe_pixel_geometry.width_m, derived_pixel_geometry.width_m, rel_tol=1e-6
                )
                and math.isclose(
                    probe_pixel_geometry.height_m, derived_pixel_geometry.height_m, rel_tol=1e-6
                )
            ):
                raise ValueError(
                    'Probe file pixel geometry disagrees with the detector-derived geometry! '
                    f'(probe={probe_pixel_geometry.width_m}x{probe_pixel_geometry.height_m} m '
                    f'derived={derived_pixel_geometry.width_m}x'
                    f'{derived_pixel_geometry.height_m} m) '
                    'Reconcile them with --detector-pixel-size-m, --detector-distance-m, '
                    'or --probe-energy-eV.'
                )

        probe_sequence = ProbeSequence(
            probe_from_file.get_array(),
            probe_from_file.get_opr_weights_or_none(),
            probe_pixel_geometry,
        )

        # Post-condition on the crop derivation above, not a user-facing check: the
        # read region was built from this very extent, so this fires only if the
        # full-frame short circuit or the bounds logic is ever changed wrongly.
        if (probe_sequence.height_px, probe_sequence.width_px) != (
            probe_geometry.height_px,
            probe_geometry.width_px,
        ):
            raise ValueError(
                'Probe file extent does not match the diffraction patterns! '
                f'(probe={probe_sequence.height_px}x{probe_sequence.width_px} '
                f'patterns={probe_geometry.height_px}x{probe_geometry.width_px})'
            )

        # Necessary but not sufficient: prepare_reconstruct_input interpolates and
        # drops positions without filtering OPR rows alongside them (its TODO), so a
        # matching count here can still drift by the time it pairs them.
        opr_weights = probe_from_file.get_opr_weights_or_none()

        if opr_weights is not None and opr_weights.shape[0] != len(positions):
            raise ValueError(
                'Probe file OPR weight count does not match the probe positions! '
                f'(opr weights={opr_weights.shape[0]} positions={len(positions)})'
            )

    object_geometry = compute_object_geometry(
        positions, probe_geometry, padding_px=args.object_padding_px
    )
    logger.info('Object geometry: %s', object_geometry)

    # Deviations = 0 gives a flat unit-amplitude, zero-phase field.
    object_ = generate_random_object(
        numpy.random.default_rng(args.seed),
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
    tomography_angle_deg = (
        0.0 if args.tomography_angle_deg is None else float(args.tomography_angle_deg)
    )

    product = Product(
        metadata=ProductMetadata(
            name='lamni-reconstruct',
            comments=f'Reconstructed from {args.diffraction_file.name}',
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=float(metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=tomography_angle_deg,
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

    # USER: customize pty-chi options here (edit fields on `options` before the loop).
    options = PtychographyTaskOptions(
        reconstructor_options=LSQMLReconstructorOptions(),
        object_options=LSQMLObjectOptions(),
        probe_options=LSQMLProbeOptions(),
        probe_position_options=LSQMLProbePositionOptions(),
        opr_mode_weight_options=LSQMLOPRModeWeightsOptions(),
    )

    options = align_task_options_with_product(options, product)
    options.check()

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)

    num_epochs = int(options.reconstructor_options.num_epochs)
    logger.info(
        'Starting reconstruction: %d patterns, %d epochs total, sync every %d',
        reconstruct_input.diffraction_patterns.shape[0],
        num_epochs,
        args.num_sync_epochs,
    )

    final_output = None
    for output in reconstruct_with_ptychi(
        reconstruct_input, options, num_sync_epochs=args.num_sync_epochs
    ):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Epoch %d/%d: loss=%.6g', output.progress, num_epochs, last_loss)
        final_output = output

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    args.output_product.parent.mkdir(parents=True, exist_ok=True)
    save_product(args.output_product, final_output.product)
    logger.info('Saved reconstructed product to %s', args.output_product)
    return 0


if __name__ == '__main__':
    sys.exit(main())
