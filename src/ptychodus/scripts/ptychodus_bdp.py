"""
Prepare experiment data for use in a beamline data pipeline
"""

import argparse
import logging
import sys
from pathlib import Path

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
        '--bad-pixels-input',
        metavar='BAD_PIXELS_INPUT_FILE',
        help='Path to bad pixel mask file',
        type=argparse.FileType('r'),
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
        '--validate-diffraction',
        action='store_true',
        help='Validate diffraction file',
    )

    args = parser.parse_args()
    diffraction_file = Path(args.diffraction_input.name)
    probe_position_file = Path(args.probe_position_input.name)
    settings_file = Path(args.settings.name)
    output_directory = args.output_directory
    bad_pixels_file = Path(args.bad_pixels_input.name) if args.bad_pixels_input else None

    logging.basicConfig(level=args.log_level)

    with ModelCore(settings_file, log_level=args.log_level) as model:
        product_api = model.workflow_api.create_product(
            args.product_name,
            detector_distance_m=args.detector_distance_m,
            probe_energy_eV=args.probe_energy_eV,
            probe_photon_count=args.probe_photon_count,
        )

        if args.defocus_distance_m is not None:
            product_api.set_probe_defocus_distance(args.defocus_distance_m)

        if args.exposure_time_s is not None:
            product_api.set_exposure_time(args.exposure_time_s)

        product_api.load_diffraction(diffraction_file)

        if args.validate_diffraction:
            return 0

        product_api.load_probe_positions(probe_position_file)

        if bad_pixels_file is not None:
            product_api.load_bad_pixels(bad_pixels_file)

        product_api.save_product(StandardFileLayout(output_directory).diffraction_file)

        if args.product_comment:
            product_api.set_product_comment(args.product_comment)

        layout = StandardFileLayout(output_directory)
        product_api.save_product(layout.product_file)

    return 0
