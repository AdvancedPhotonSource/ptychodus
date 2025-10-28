#!/usr/bin/env python
"""
Convert PtychoShelves datasets to Ptychodus format

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
from typing import Any, Final
import argparse
import json
import logging
import sys

from ptychodus.api.plugins import PluginChooser
from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def version_string() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def list_plugin_names(plugin_chooser: PluginChooser[Any]) -> str:
    return ', '.join(sorted(plugin.simple_name for plugin in plugin_chooser))


def main() -> int:
    prog = Path(__file__).stem.lower()
    parser = argparse.ArgumentParser(
        prog=prog, description=f'{prog} repackages ptychoshelves datasets in ptychodus formats.'
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
        '--override-object',
        metavar='OBJECT_FILE',
        type=argparse.FileType('r'),
        help='Path to the object file.',
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
        '--product-comment',
        default='',
        help='Data product comment',
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
        required=True,
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
        version=version_string(),
    )

    args = parser.parse_args()
    settings_file = Path(args.settings.name) if args.settings else None

    DIFFRACTION_FILE_TYPE: Final[str] = 'PtychoShelves'  # noqa: N806
    PRODUCT_FILE_TYPE: Final[str] = 'PtychoShelves'  # noqa: N806

    with ModelCore(settings_file) as model:
        if args.list_plugins:
            plugins: dict[str, Sequence[str]] = dict()
            plugins['diffraction_readers'] = list_plugin_names(
                model.plugin_registry.diffraction_file_readers
            )
            plugins['product_readers'] = list_plugin_names(
                model.plugin_registry.product_file_readers
            )
            plugins['probe_position_readers'] = list_plugin_names(
                model.plugin_registry.probe_position_file_readers
            )
            plugins['probe_readers'] = list_plugin_names(model.plugin_registry.probe_file_readers)
            plugins['object_readers'] = list_plugin_names(model.plugin_registry.object_file_readers)

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
            file_type=DIFFRACTION_FILE_TYPE,
            process_patterns=False,
            block=True,
        )
        model.workflow_api.export_assembled_patterns(args.diffraction_output.name)
        logger.info(f'Wrote diffraction data to {args.diffraction_output.name}')

        product_api = model.workflow_api.open_product(
            Path(args.product_input.name), file_type=PRODUCT_FILE_TYPE
        )

        if args.rename_product is not None:
            product_api.rename_product(args.rename_product)

        if args.override_object is not None:
            product_api.open_object(Path(args.override_object.name))

        if args.override_probe is not None:
            product_api.open_probe(Path(args.override_probe.name))

        if args.override_probe_positions is not None:
            product_api.open_probe_positions(Path(args.override_probe_positions.name))

        product_api.save_product(Path(args.product_output.name), file_type='HDF5')
        logger.info(f'Wrote product data to {args.product_output.name}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
