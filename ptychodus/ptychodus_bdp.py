#!/usr/bin/env python

from pathlib import Path
import argparse
import sys

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import CropCenter
from ptychodus.api.settings import PathPrefixChange
from ptychodus.model import ModelCore
import ptychodus


def versionString() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


class DirectoryType:

    def __init__(self, *, must_exist: bool) -> None:
        self._must_exist = must_exist

    def __call__(self, string: str) -> Path:
        path = Path(string)

        if self._must_exist and not path.is_dir():
            raise argparse.ArgumentTypeError(f'\"{string}\" is not a directory!')

        return path


def main() -> int:
    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f'{prog} prepares experiment data for use in beamline data pipelines',
    )
    parser.add_argument(
        '-c',
        '--comment',
        help='data product comment',
    )
    parser.add_argument(
        '-d',
        '--dev',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-n',
        '--name',
        help='data product name',
        required=True,
    )
    parser.add_argument(
        '-o',
        '--output_directory',
        metavar='OUTPUT_DIR',
        type=DirectoryType(must_exist=False),
        required=True,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        help='use default settings from file',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--patterns_file_path',
        metavar='PATTERNS_FILE',
        help='diffraction patterns file path',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--crop_center_x_px',
        metavar='CENTER_X',
        help='crop center x in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop_center_y_px',
        metavar='CENTER_Y',
        help='crop center y in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop_width_px',
        metavar='WIDTH',
        help='crop width in pixels',
        type=int,
    )
    parser.add_argument(
        '--crop_height_px',
        metavar='HEIGHT',
        help='crop height in pixels',
        type=int,
    )
    parser.add_argument(
        '--scan_file_path',
        metavar='SCAN_FILE',
        help='scan file path',
        type=argparse.FileType('r'),
        required=True,
    )
    parser.add_argument(
        '--number_of_gpus',
        metavar='INTEGER',
        help='number of GPUs to use in reconstruction',
        type=int,
    )
    parser.add_argument(
        '--probe_energy_eV',
        metavar='ENERGY',
        help='probe energy in electron volts',
        type=float,
    )
    parser.add_argument(
        '--probe_photon_flux_Hz',
        metavar='FLUX',
        help='probe number of photons per second',
        type=float,
    )
    parser.add_argument(
        '--exposure_time_s',
        metavar='TIME',
        help='exposure time in seconds',
        type=float,
    )
    parser.add_argument(
        '--detector_distance_m',
        metavar='DISTANCE',
        help='detector distance in meters',
        type=float,
    )
    parser.add_argument(
        '--local_path_prefix',
        metavar='PATH_PREFIX',
        help='local posix path prefix',
        type=DirectoryType(must_exist=True),
    )
    parser.add_argument(
        '--remote_path_prefix',
        metavar='PATH_PREFIX',
        help='remote posix path prefix',
        type=DirectoryType(must_exist=False),
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=versionString(),
    )

    args = parser.parse_args()
    changePathPrefix: PathPrefixChange | None = None
    cropCenter: CropCenter | None = None
    cropExtent: ImageExtent | None = None

    print(args)  # FIXME

    if bool(args.local_path_prefix) ^ bool(args.remote_path_prefix):
        parser.error('--local_path_prefix and --remote_path_prefix'
                     'must be given together.')
    else:
        changePathPrefix = PathPrefixChange(
            findPathPrefix=args.local_path_prefix,
            replacementPathPrefix=args.remote_path_prefix,
        )

    if bool(args.crop_center_x_px) ^ bool(args.crop_center_y_px):
        parser.error('--crop_center_x_px and --crop_center_y_px must be given together.')
    # FIXME args.crop_center_x_px
    # FIXME args.crop_center_y_px

    if bool(args.crop_width_px) ^ bool(args.crop_height_px):
        parser.error('--crop_width_px and --crop_height_px must be given together.')
    # FIXME args.crop_width_px
    # FIXME args.crop_height_px

    with ModelCore(Path(args.settings), isDeveloperModeEnabled=args.dev) as model:
        model.workflowAPI.openPatterns(args.patterns_file_path,
                                       cropCenter=cropCenter,
                                       cropExtent=cropExtent)
        workflowProductAPI = model.workflowAPI.createProduct(
            name=args.name,
            comments=args.comments,
            detectorDistanceInMeters=args.detector_distance_m,
            probeEnergyInElectronVolts=args.probe_energy_eV,
            probePhotonsPerSecond=args.probe_photon_flux_Hz,
            exposureTimeInSeconds=args.exposure_time_s,
        )
        workflowProductAPI.openScan(args.scan_file_path)
        workflowProductAPI.syncProductToSettings()

        # FIXME args.number_of_gpus

        stagingDir = args.output_directory
        stagingDir.mkdir(parents=True, exist_ok=True)
        # FIXME settingsRegistry.saveSettings(stagingDir / 'settings.ini', changePathPrefix)
        # FIXME patternsAPI.exportPreprocessedPatterns(stagingDir / 'patterns.npz')
        workflowProductAPI.saveProduct(stagingDir / 'product-in.npz', fileType='NPZ')

    return 0


sys.exit(main())
