"""
Convert prepared experiment data to ptychodus format

Usage:
    convert-to-ptychodus \
        --diffraction-input data/fly001.mat \
        --diffraction-output data/fly001_diffraction.h5 \
        --probe-positions-input data/fly001_positions.csv \
        --product-input data/fly001_probe.npy data/fly001_object.npy \
        --product-output "data/fly001_product.h5" \
        --product-name "fly001" \
        --settings ptychodus.ini

    convert-to-ptychodus \
        --product-input "data/velo_19c2_Jun_IC_fly145_product.h5" \
        --product-output "data/velo_19c2_Jun_IC_fly145_product.h5" \
        --product-name "velo_19c2_Jun_IC_fly145"
"""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def main() -> int:
    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog, description=f'{prog} repackages prepared datasets into ptychodus formats.'
    )
    parser.add_argument(
        '--diffraction-input-type',
        default='fold_slice',
        help='Diffraction input file type.',
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        type=argparse.FileType('r'),
        help='Path to the diffraction input file.',
    )
    parser.add_argument(
        '--diffraction-output',
        metavar='DIFFRACTION_OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='Path to the diffraction output file.',
    )
    parser.add_argument(
        '--list-plugins',
        action='store_true',
        help='List available file reader plugins, then exit.',
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.WARNING,
        help='Set Python logging level.',
    )
    parser.add_argument(
        '--override-object-type',
        metavar='OBJECT_FILE_TYPE',
        default='fold_slice',
        help='Override object file type.',
    )
    parser.add_argument(
        '--override-object',
        metavar='OBJECT_FILE',
        type=argparse.FileType('r'),
        help='Path to the object file.',
    )
    parser.add_argument(
        '--override-probe-type',
        metavar='PROBE_FILE_TYPE',
        default='fold_slice',
        help='Override probe file type.',
    )
    parser.add_argument(
        '--override-probe',
        metavar='PROBE_FILE',
        type=argparse.FileType('r'),
        help='Path to the probe file.',
    )
    parser.add_argument(
        '--override-probe-positions-type',
        metavar='PROBE_POSITIONS_FILE_TYPE',
        default='fold_slice',
        help='Override probe positions file type.',
    )
    parser.add_argument(
        '--override-probe-positions',
        metavar='PROBE_POSITIONS_FILE',
        type=argparse.FileType('r'),
        help='Path to the probe positions file.',
    )
    parser.add_argument(
        '--product-input-type',
        default='fold_slice',
        help='Product input file type.',
    )
    parser.add_argument(
        '--product-input',
        metavar='PRODUCT_INPUT_FILE',
        type=argparse.FileType('r'),
        help='Path to the product input file.',
    )
    parser.add_argument(
        '--product-name',
        help='Data product name',
    )
    parser.add_argument(
        '--product-output',
        metavar='PRODUCT_OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='Path to the product output file.',
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
        version=ptychodus.VERSION_STRING,
    )

    args = parser.parse_args()
    settings_file = Path(args.settings.name) if args.settings else None

    logging.basicConfig(level=args.log_level)

    if args.list_plugins:
        with ModelCore(settings_file) as model:
            for name, display_name in model.diffraction_core.file_reader_registry.items():
                logger.info(f'{name}: {display_name}')
        return 0

    with ModelCore(settings_file, log_level=args.log_level) as model:
        product_api = model.workflow_api.create_product(
            args.product_name or 'converted',
        )

        if args.diffraction_input is not None:
            diffraction_file = Path(args.diffraction_input.name)
            product_api.load_diffraction(diffraction_file, file_type=args.diffraction_input_type)

        if args.override_probe is not None:
            probe_file = Path(args.override_probe.name)
            product_api.load_probe(probe_file, file_type=args.override_probe_type)

        if args.override_probe_positions is not None:
            probe_positions_file = Path(args.override_probe_positions.name)
            product_api.load_probe_positions(
                probe_positions_file, file_type=args.override_probe_positions_type
            )

        if args.override_object is not None:
            object_file = Path(args.override_object.name)
            product_api.load_object(object_file, file_type=args.override_object_type)

        if args.product_input is not None:
            product_file = Path(args.product_input.name)
            product_api.load_product(product_file, file_type=args.product_input_type)

        if args.diffraction_output is not None:
            diffraction_output_file = Path(args.diffraction_output.name)
            product_api.save_diffraction(diffraction_output_file)

        if args.product_output is not None:
            product_output_file = Path(args.product_output.name)
            product_api.save_product(product_output_file)

    return 0
