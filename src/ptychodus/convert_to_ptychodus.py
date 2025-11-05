#!/usr/bin/env python
"""
Convert prepared datasets to ptychodus format

Example usage:

convert-to-ptychodus \
    --diffraction-input "data/ptychoshelves/velo_19c2_Jun_IC_fly145/data_roi0_dp.hdf5" \
    --diffraction-output "data/velo_19c2_Jun_IC_fly145_diffraction.h5" \
    --product-input "data/ptychoshelves/velo_19c2_Jun_IC_fly145/Niter100.mat" \
    --product-output "data/velo_19c2_Jun_IC_fly145_product.h5" \
    --product-name "velo_19c2_Jun_IC_fly145"
"""

from collections.abc import Sequence
from pathlib import Path
import argparse
import json
import logging
import sys

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

    with ModelCore(settings_file, log_level=args.log_level) as model:
        if args.list_plugins:
            plugins: dict[str, Sequence[str]] = dict()
            registry = model.plugin_registry
            plugins['diffraction_readers'] = (
                registry.diffraction_file_readers.stringify_plugin_names()
            )
            plugins['product_readers'] = registry.product_file_readers.stringify_plugin_names()
            plugins['probe_position_readers'] = (
                registry.probe_position_file_readers.stringify_plugin_names()
            )
            plugins['probe_readers'] = registry.probe_file_readers.stringify_plugin_names()
            plugins['object_readers'] = registry.object_file_readers.stringify_plugin_names()

            print(json.dumps(plugins, indent=4))

            return 0

        missing_args: list[str] = []

        if args.diffraction_input is None:
            missing_args.append('--diffraction-input')

        if args.diffraction_output is None:
            missing_args.append('--diffraction-output')

        if args.product_input is None:
            missing_args.append('--product-input')

        if args.product_output is None:
            missing_args.append('--product-output')

        if missing_args:
            missing_args_str = ', '.join(missing_args)
            parser.error(f'the following arguments are required: {missing_args_str}')

        model.workflow_api.open_patterns(
            Path(args.diffraction_input.name),
            file_type=args.diffraction_input_type,
            process_patterns=False,
            block=True,
        )
        model.workflow_api.export_assembled_patterns(args.diffraction_output.name)
        logger.info(f'Wrote diffraction data to {args.diffraction_output.name}')

        product_api = model.workflow_api.open_product(
            Path(args.product_input.name),
            file_type=args.product_input_type,
        )

        if args.product_name is not None:
            product_api.rename_product(args.product_name)

        if args.override_object is not None:
            product_api.open_object(
                Path(args.override_object.name), file_type=args.override_object_type
            )

        if args.override_probe is not None:
            product_api.open_probe(
                Path(args.override_probe.name), file_type=args.override_probe_type
            )

        if args.override_probe_positions is not None:
            product_api.open_probe_positions(
                Path(args.override_probe_positions.name),
                file_type=args.override_probe_positions_type,
            )

        product_api.save_product(Path(args.product_output.name), file_type='HDF5')
        logger.info(f'Wrote product data to {args.product_output.name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
