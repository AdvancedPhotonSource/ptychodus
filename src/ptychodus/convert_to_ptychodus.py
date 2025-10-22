#!/usr/bin/env python
"""
Convert ptychography data format using Ptychodus.

Example usage:
python convert_format.py \
    --patterns "data/ptychoshelves/velo_19c2_Jun_IC_fly145/data_roi0_dp.hdf5" \
    --probe "data/ptychoshelves/velo_19c2_Jun_IC_fly145/Niter100.mat" \
    --probe-positions "data/ptychoshelves/velo_19c2_Jun_IC_fly145/data_roi0_para.hdf5" \
    --metadata "data/ptychoshelves/velo_19c2_Jun_IC_fly145/data_roi0_para.hdf5" \
    --product-name "ptychodus" \
    --output-dir "outputs" \

"""
# FIXME update example usage

from pathlib import Path
import argparse
import logging
import sys
from typing import Final

from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import ImageExtent
from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def version_string() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Convert PtychoShelves reconstructions to the Ptychodus HDF5 format.'
    )
    parser.add_argument(
        '-d',
        '--dev',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        type=argparse.FileType('r'),
        help='Path to the diffraction input file.',
        required=True,
    )
    parser.add_argument(
        '--diffraction-output',
        metavar='DIFFRACTION_OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='Path to the diffraction output file.',
        required=True,
    )
    parser.add_argument(
        '--list-plugins',
        action='store_true',
        help='List available diffraction and scan plugins, then exit.',
    )
    parser.add_argument(
        '--override-probe',
        metavar='PROBE_FILE',
        type=argparse.FileType('r'),
        help='Path to the probe file.',
    )
    parser.add_argument(
        '--override-probe-positions',
        metavar='PROBE_POSITIONS_FILE',
        type=argparse.FileType('r'),
        help='Path to the probe positions file.',
    )
    parser.add_argument(
        '--product-input',
        metavar='PRODUCT_INPUT_FILE',
        type=argparse.FileType('r'),
        help='Path to the product input file.',
        required=True,
    )
    parser.add_argument(
        '--product-output',
        metavar='PRODUCT_OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='Path to the product output file.',
        required=True,
    )
    parser.add_argument(
        '--rename-product',
        help='Changes the data product name.',
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        type=argparse.FileType('r'),
        help='Path to the settings file.',
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=version_string(),
    )

    args = parser.parse_args()
    settings_file = Path(args.settings.name) if args.settings else None

    DIFFRACTION_FILE_TYPE: Final[str] = 'PtychoShelves'  # noqa: N806
    PRODUCT_FILE_TYPE: Final[str] = 'PtychoShelves'  # noqa: N806

    with ModelCore(settings_file, is_developer_mode_enabled=args.dev) as model:
        if args.list_plugins:
            # FIXME print('Diffraction readers   : ', ', '.join(sorted(DiffractionReaderChoices)))
            # FIXME print('Diffraction writers   : ', ', '.join(sorted(DiffractionWriterChoices)))
            # FIXME print('Probe readers         : ', ', '.join(sorted(ProbeReaderChoices)))
            # FIXME print('Probe position readers: ', ', '.join(sorted(ProbePositionReaderChoices)))
            # FIXME print('Product writers       : ', ', '.join(sorted(ProductWriterChoices)))
            return 0

        dp_size = (0, 0)  # FIXME

        # FIXME support cropping or disable cropping
        model.workflow_api.open_patterns(
            args.diffraction_input.name,
            file_type=DIFFRACTION_FILE_TYPE,
            crop_center=CropCenter(position_x_px=dp_size[1] // 2, position_y_px=dp_size[0] // 2),
            crop_extent=ImageExtent(width_px=dp_size[1], height_px=dp_size[0]),
        )
        # FIXME wait for patterns to load?
        model.workflow_api.export_assembled_patterns(args.diffraction_output.name)
        logger.info(f'Wrote diffraction data to {args.diffraction_output.name}')

        product_api = model.workflow_api.open_product(
            args.product_input.name, file_type=PRODUCT_FILE_TYPE
        )

        # FIXME optionally rename product

        if args.override_probe is not None:
            product_api.open_probe(Path(args.override_probe.name))

        if args.override_probe_positions is not None:
            product_api.open_probe_positions(Path(args.override_probe_positions.name))

        product_api.save_product(args.product_output.name, file_type='HDF5')
        logger.info(f'Wrote product data to {args.product_output.name}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
