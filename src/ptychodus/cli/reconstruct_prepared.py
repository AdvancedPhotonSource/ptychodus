#!/usr/bin/env python
"""Reconstruct a prepared directory with pty-chi.

The second half of a two-stage run. A standard driver given ``--prepare-only`` reads the
beamline files, assembles the patterns and builds an initial product, then writes them to a
prepared directory with the pty-chi options aligned against that product. This command
picks the run up from there: it reads no raw files and knows nothing about any instrument,
so the same prepared directory can be reconstructed on another node, or again with other
options, without assembling the data a second time.

The prepared directory is left untouched. Its ``product.h5`` is the initial product and the
output's ``product.h5`` is the final one, which is why the two directories must differ.

``ptychodus-reconstruct-subprocess`` reconstructs the same input for a launching process,
reporting progress as an event stream; this command logs to the console instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout, load_diffraction_data, load_product
from ptychodus.api.reconstruct import PositionIndexFilter
from ptychodus.cli import (
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    positive_int,
)
from ptychodus.cli._reconstruct_common import (
    add_print_ptychi_options_argument,
    add_ptychi_options_argument,
    install_signal_handlers,
    load_ptychi_options,
    pair_patterns_with_positions,
    resolve_ptychi_options_file,
    run_reconstruction,
)

logger = logging.getLogger('reconstruct_prepared')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Reconstruct a directory prepared by a driver run with --prepare-only.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=True),
        help='Prepared directory holding diffraction.h5, product.h5 and optionally '
        'ptychi_options.yaml. It is not modified.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory, written in the ptychodus standard layout: '
            'ptychi_options.yaml, per-epoch product.NNNNNN.h5 checkpoints, and the final '
            'product.h5. Must differ from INPUT_DIR.'
        ),
    )
    add_ptychi_options_argument(
        parser,
        default_note=(
            'Defaults to the one in INPUT_DIR, else to stock LSQML, or BH for a near-field product.'
        ),
    )
    add_print_ptychi_options_argument(parser)
    parser.add_argument(
        '--index-filter',
        choices=tuple(member.name.lower() for member in PositionIndexFilter),
        default=PositionIndexFilter.ALL.name.lower(),
        help='Scan index subset to reconstruct.',
    )
    parser.add_argument(
        '--num-sync-epochs',
        type=positive_int,
        default=100,
        help='Epochs between checkpoints, progress logs and cancellation checks.',
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

    input_directory: Path = args.input_directory
    output_directory: Path = args.output_directory

    if output_directory.resolve() == input_directory.resolve():
        parser.error(
            'OUTPUT_DIR must differ from INPUT_DIR: the final product.h5 would overwrite '
            'the initial one.'
        )

    diffraction_file = StandardFileLayout.DIFFRACTION.path(input_directory)
    input_product_file = StandardFileLayout.PRODUCT.path(input_directory)

    for required_file in (diffraction_file, input_product_file):
        if not required_file.is_file():
            parser.error(
                f'{input_directory} holds no {required_file.name}; is it a prepared directory?'
            )

    configure_logging(args)

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored.
    cancellation = install_signal_handlers()

    # A named or staged file is decoded before the data are read, so options that do not
    # parse fail in seconds rather than after loading several gigabytes.
    options_file = resolve_ptychi_options_file(logger, args.ptychi_options, input_directory)
    options = None if options_file is None else load_ptychi_options(options_file)

    logger.info('Reading %s', diffraction_file)
    assembled_data = load_diffraction_data(diffraction_file, mmap_file=args.mmap_file)
    logger.info('Reading %s', input_product_file)
    product = load_product(input_product_file)

    if options is None:
        # The stock algorithm follows the propagation regime, which only the product states.
        options = load_ptychi_options(None, far_field=product.metadata.far_field)

    reconstruct_input = pair_patterns_with_positions(
        logger,
        assembled_data,
        product,
        index_filter=PositionIndexFilter[args.index_filter.upper()],
    )
    num_epochs = int(options.reconstructor_options.num_epochs)

    if args.dry_run:
        logger.info(
            'Dry run: %d patterns, %d epochs (%s), probe %s, object %s.',
            reconstruct_input.diffraction_patterns.shape[0],
            num_epochs,
            type(options).__name__,
            reconstruct_input.product.probes.get_array().shape,
            reconstruct_input.product.object_.get_array().shape,
        )
        logger.info('Would write %s', StandardFileLayout.PRODUCT.path(output_directory))
        logger.info(
            'Would write checkpoints like %s',
            StandardFileLayout.PRODUCT.checkpoint_path(output_directory, args.num_sync_epochs),
        )
        logger.info(
            'Would write %s once the options are aligned',
            StandardFileLayout.PTYCHI_OPTIONS.path(output_directory),
        )
        logger.info('Nothing written: stopping before the pty-chi option alignment.')
        return ExitCode.SUCCESS

    output_directory.mkdir(parents=True, exist_ok=True)
    run_reconstruction(
        logger,
        reconstruct_input,
        options,
        output_directory,
        num_sync_epochs=args.num_sync_epochs,
        cancellation=cancellation,
    )
    return ExitCode.CANCELLED if cancellation.is_cancelled else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
