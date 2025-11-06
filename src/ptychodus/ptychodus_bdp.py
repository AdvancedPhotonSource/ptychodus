#!/usr/bin/env python
"""
Prepare experiment data for use in a beamline data pipeline
"""

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import StandardFileLayout
from ptychodus.cli import DirectoryType
from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def main() -> int:
    crop_center: CropCenter | None = None
    crop_extent: ImageExtent | None = None

    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f'{prog} prepares experiment data for use in beamline data pipelines',
    )
    parser.add_argument(
        '--crop-center-x-px',
        metavar='CENTER_X',
        help='Diffraction pattern crop center x in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop-center-y-px',
        metavar='CENTER_Y',
        help='Diffraction pattern crop center y in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop-width-px',
        metavar='WIDTH',
        help='Diffraction pattern crop width in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop-height-px',
        metavar='HEIGHT',
        help='Diffraction pattern crop height in pixels',
        type=int,
    )
    parser.add_argument(
        '--defocus-distance-m',
        metavar='DISTANCE',
        help='Defocus distance in meters',
        type=float,
    )
    parser.add_argument(
        '--detector-distance-m',
        metavar='DISTANCE',
        help='Detector distance in meters',
        type=float,
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        help='Path to diffraction file.',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--exposure-time-s',
        metavar='TIME',
        help='Exposure time in seconds',
        type=float,
    )
    parser.add_argument(
        '--log-level',
        default=logging.INFO,
        help='Python logging level.',
        type=int,
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        type=DirectoryType(must_exist=False),
        required=True,
    )
    parser.add_argument(
        '--probe-energy-eV',
        metavar='ENERGY',
        help='Probe energy in electron volts',
        type=float,
    )
    parser.add_argument(
        '--probe-photon-count',
        metavar='NUMBER',
        help='Probe number of photons',
        type=float,
    )
    parser.add_argument(
        '--probe-position-input',
        metavar='PROBE_POSITION_INPUT_FILE',
        help='Path to probe position input file',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--product-comment',
        default='',
        help='Data product comment',
    )
    parser.add_argument(
        '--product-name',
        help='Data product name',
        required=True,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        help='Use default settings from file',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--tomography-angle-deg',
        metavar='ANGLE',
        help='Tomography angle in degrees',
        type=float,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )

    args = parser.parse_args()

    if args.crop_center_x_px is not None and args.crop_center_y_px is not None:
        crop_center = CropCenter(
            position_x_px=args.crop_center_x_px,
            position_y_px=args.crop_center_y_px,
        )
    elif bool(args.crop_center_x_px) ^ bool(args.crop_center_y_px):
        parser.error('--crop-center-x-px and --crop-center-y-px must be given together.')

    if args.crop_width_px is not None and args.crop_height_px is not None:
        crop_extent = ImageExtent(
            width_px=args.crop_width_px,
            height_px=args.crop_height_px,
        )
    elif bool(args.crop_width_px) ^ bool(args.crop_height_px):
        parser.error('--crop-width-px and --crop-height-px must be given together.')

    if args.defocus_distance_m is not None:
        logger.warning('Defocus distance is not implemented yet!')  # TODO

    with ModelCore(Path(args.settings.name), log_level=args.log_level) as model:
        model.workflow_api.open_patterns(
            Path(args.diffraction_input.name),
            crop_center=crop_center,
            crop_extent=crop_extent,
            block=True,
        )
        workflow_product_api = model.workflow_api.create_product(
            name=args.product_name,
            comments=args.product_comment,
            detector_distance_m=args.detector_distance_m,
            probe_energy_eV=args.probe_energy_eV,
            probe_photon_count=args.probe_photon_count,
            exposure_time_s=args.exposure_time_s,
            tomography_angle_deg=args.tomography_angle_deg,
        )
        workflow_product_api.open_probe_positions(Path(args.probe_position_input.name))
        workflow_product_api.generate_probe()
        workflow_product_api.generate_object()

        staging_dir = args.output_directory
        staging_dir.mkdir(parents=True, exist_ok=True)
        model.workflow_api.save_settings(staging_dir / StandardFileLayout.SETTINGS)
        model.workflow_api.export_assembled_patterns(staging_dir / StandardFileLayout.DIFFRACTION)
        workflow_product_api.save_product(
            staging_dir / StandardFileLayout.PRODUCT_IN, file_type='HDF5'
        )

    return 0


if __name__ == '__main__':
    sys.exit(main())
