#!/usr/bin/env python
"""Reconstruct a prepared directory with any installed reconstructor.

The second half of a two-stage run. A standard driver given ``--prepare-only`` reads the
beamline files, assembles the patterns and builds an initial product, then writes them to a
prepared directory with the options it would have run. This command picks the run up from
there: it reads no raw files and knows nothing about any instrument, so the same prepared
directory can be reconstructed on another node, or again with other options -- or another
library -- without assembling the data a second time.

The options are a document naming the library and algorithm beside the backend's own
options (``--print-options`` writes one), and the run goes through that backend's plugin in
this process. A trainable backend loads the ``model`` file the input directory holds.

The prepared directory is left untouched. Its ``product.h5`` is the initial product and the
output's ``product.h5`` is the final one, which is why the two directories must differ.

A scan in a storage root -- as ``ptychodus-stream`` saves them -- is named instead by its
product uuid, with ``--product``; the storage layout finds its diffraction data through the
product's lineage.

``ptychodus-reconstruct-subprocess`` reconstructs the same input for a launching process,
reporting progress as an event stream; this command logs to the console instead.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from uuid import UUID

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout, load_diffraction_data, load_product
from ptychodus.api.reconstruct import PositionIndexFilter
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    LayoutError,
    ManifestLoadError,
    StorageLayout,
    resolve_storage_inputs,
)
from ptychodus.api.options import OptionsError
from ptychodus.api.plugins import PluginRegistry
from ptychodus.cli import (
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    open_action_log,
)
from ptychodus.cli._reconstruct_common import (
    add_options_argument,
    add_print_options_argument,
    default_reconstruction,
    find_options_document,
    install_signal_handlers,
    load_trained_model,
    pair_patterns_with_positions,
    read_reconstruction,
    run_reconstruction,
)

logger = logging.getLogger('reconstruct_prepared')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Reconstruct a directory prepared by a driver run with --prepare-only.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        type=DirectoryType(must_exist=True),
        help='Prepared directory holding diffraction.h5, product.h5 and optionally '
        'reconstruct_options.yaml and a trained model. It is not modified.',
    )
    source.add_argument(
        '--product',
        metavar='UUID',
        type=UUID,
        help='Uuid of a product in the storage root, reconstructed with the diffraction '
        'data it derives from.',
    )
    parser.add_argument(
        '--storage-root',
        metavar='STORAGE_ROOT',
        type=Path,
        default=os.environ.get(STORAGE_ROOT_ENVIRONMENT_VARIABLE),
        help=f'Storage root holding --product. Defaults to ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory, written in the ptychodus standard layout: '
            'reconstruct_options.yaml, reconstruct.log, product.NNNNNN.h5 checkpoints, and '
            'the final product.h5. Must differ from INPUT_DIR.'
        ),
    )
    add_options_argument(
        parser,
        default_note=(
            'Defaults to the one in INPUT_DIR, else to stock pty-chi LSQML, or BH for a '
            'near-field product.'
        ),
    )
    add_print_options_argument(parser)
    parser.add_argument(
        '--index-filter',
        choices=tuple(member.name.lower() for member in PositionIndexFilter),
        default=PositionIndexFilter.ALL.name.lower(),
        help='Scan index subset to reconstruct.',
    )
    parser.add_argument(
        '--mmap-file',
        metavar='MMAP_FILE',
        type=Path,
        help='Stage diffraction patterns into this memmap instead of loading into RAM.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load and pair the inputs and report the configuration without reconstructing.',
    )
    add_log_level_argument(parser)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def main() -> ExitCode:
    parser = _build_parser()
    args = parser.parse_args()

    input_directory: Path | None = args.input_directory
    output_directory: Path = args.output_directory

    if input_directory is None:
        if args.storage_root is None:
            parser.error(f'--product needs --storage-root or ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.')

        try:
            diffraction_file, input_product_file = resolve_storage_inputs(
                StorageLayout(Path(args.storage_root)), args.product
            )
        except (LayoutError, ManifestLoadError) as exc:
            parser.error(str(exc))
    else:
        diffraction_file = StandardFileLayout.DIFFRACTION.path(input_directory)
        input_product_file = StandardFileLayout.PRODUCT.path(input_directory)

        for required_file in (diffraction_file, input_product_file):
            if not required_file.is_file():
                parser.error(
                    f'{input_directory} holds no {required_file.name}; is it a prepared directory?'
                )

    if output_directory.resolve() == input_product_file.parent.resolve():
        parser.error(
            'OUTPUT_DIR must differ from the input: the final product.h5 would overwrite '
            'the initial one.'
        )

    configure_logging(args)

    if not args.dry_run:
        open_action_log(output_directory, StandardFileLayout.RECONSTRUCT_LOG)

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored.
    cancellation = install_signal_handlers()
    libraries = PluginRegistry.load_plugins().reconstructor_libraries

    # A named or staged document is read before the data are, so options that do not
    # parse fail in seconds rather than after loading several gigabytes. Without one, the
    # stock algorithm follows the propagation regime, which only the product states.
    options_file = (
        args.options
        if input_directory is None
        else find_options_document(args.options, input_directory)
    )

    try:
        resolved = None if options_file is None else read_reconstruction(libraries, options_file)
    except OptionsError as exc:
        logger.error('%s', exc)
        return ExitCode.USAGE

    logger.info('Reading %s', diffraction_file)
    assembled_data = load_diffraction_data(diffraction_file, mmap_file=args.mmap_file)
    logger.info('Reading %s', input_product_file)
    product = load_product(input_product_file)

    if resolved is None:
        try:
            resolved = default_reconstruction(libraries, far_field=product.metadata.far_field)
        except OptionsError as exc:
            logger.error('%s', exc)
            return ExitCode.USAGE

    reconstruct_input = pair_patterns_with_positions(
        logger,
        assembled_data,
        product,
        index_filter=PositionIndexFilter[args.index_filter.upper()],
    )

    if args.dry_run:
        logger.info(
            'Dry run: %d patterns, %s %s, probe %s, object %s.',
            reconstruct_input.diffraction_patterns.shape[0],
            resolved.library,
            resolved.reconstructor.name,
            reconstruct_input.product.probes.get_array().shape,
            reconstruct_input.product.object_.get_array().shape,
        )
        logger.info('Would write %s', StandardFileLayout.PRODUCT.path(output_directory))
        logger.info('Would write %s', StandardFileLayout.RECONSTRUCT_OPTIONS.path(output_directory))
        logger.info('Nothing written.')
        return ExitCode.SUCCESS

    if input_directory is not None:
        load_trained_model(logger, resolved.reconstructor, input_directory)

    output_directory.mkdir(parents=True, exist_ok=True)
    run_reconstruction(
        logger, resolved, reconstruct_input, output_directory, cancellation=cancellation
    )
    return ExitCode.CANCELLED if cancellation.is_cancelled else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
