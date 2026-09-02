#!/usr/bin/env python
"""Reconstruct one APS 31-ID-E LamNI ptychography dataset through the ptychodus api."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy

from ptychi.api import LSQMLOptions

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import CropRegion
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import save_product
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import CropStep, DiffractionPrepPipeline
from ptychodus.api.probe import ProbeGeometry, ProbeSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.simulate.probe import generate_fresnel_zone_plate_probe
from ptychodus.model.ptychi.task import reconstruct_with_ptychi

logger = logging.getLogger('lamni_reconstruct')


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
        help='LamNI Orchestra probe-position .dat file.',
    )
    parser.add_argument(
        '--output-product',
        required=True,
        type=Path,
        help='Destination path for the reconstructed product HDF5.',
    )
    parser.add_argument(
        '--crop-extent-px',
        type=int,
        default=None,
        help='Square crop side in raw detector pixels; uses the HDF5 beam center. Omit to skip.',
    )
    parser.add_argument(
        '--fzp-preset',
        default='APS 31-ID-E LamNI',
        help='Name of the FresnelZonePlate plugin preset to use.',
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=800e-6,
        help='Defocus distance from the FZP focal plane; default matches the LYNX config.',
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
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(
        'APS_LamNI_Orchestra'
    )
    zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)

    logger.info('Reading raw diffraction from %s', args.diffraction_file)
    raw_dataset = diffraction_reader.read(args.diffraction_file)
    metadata = raw_dataset.get_metadata()

    if metadata.detector_distance_m is None:
        raise ValueError('Detector distance is missing from the HDF5 metadata.')
    if metadata.probe_energy_eV is None:
        raise ValueError('Probe energy is missing from the HDF5 metadata.')
    if metadata.detector_pixel_geometry is None:
        raise ValueError('Detector pixel geometry is missing from the HDF5 metadata.')

    detector_distance_m = metadata.detector_distance_m
    probe_energy_eV = metadata.probe_energy_eV  # noqa: N806

    pipeline: DiffractionPrepPipeline | None = None
    if args.crop_extent_px is not None:
        if metadata.crop_center is None:
            raise ValueError('--crop-extent-px was set but the HDF5 provides no beam center.')
        pipeline = DiffractionPrepPipeline(
            steps=(
                CropStep(
                    region=CropRegion.from_center_extent(
                        metadata.crop_center,
                        ImageExtent(width_px=args.crop_extent_px, height_px=args.crop_extent_px),
                    ),
                ),
            )
        )

    logger.info('Assembling diffraction patterns')
    assembled_data = assemble_dataset(raw_dataset, pipeline)

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

    probe = generate_fresnel_zone_plate_probe(
        probe_geometry,
        zone_plate,
        probe_wavelength_m=probe_wavelength_m,
        defocus_distance_m=args.fzp_defocus_m,
    )
    probe_sequence = ProbeSequence.from_probe(probe)

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
    options = LSQMLOptions()

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
