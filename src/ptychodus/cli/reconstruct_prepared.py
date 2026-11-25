#!/usr/bin/env python
"""Reconstruct a prepared directory with any installed reconstructor.

The second half of a two-stage run. A standard driver given ``--prepare-only`` reads the
beamline files, assembles the patterns and builds an initial product, then writes them to a
prepared directory with the options it would have run. This command picks the run up from
there: it reads no raw files and knows nothing about any instrument, so the same prepared
directory can be reconstructed on another node, or again with other options -- or another
library -- without assembling the data a second time.

The options are a document naming the library and algorithm beside the library's own
options (``--print-options`` writes one), and the run goes through that library's plugin in
this process. A trainable reconstructor loads the ``model`` file the input directory holds.

The prepared directory is left untouched. Its ``product.h5`` is the initial product and the
output's ``product.h5`` is the final one, which is why the two directories must differ.

A scan in a storage root -- as ``ptychodus-stream`` saves them -- is named instead by its
product uuid, with ``--product``; the storage layout finds its diffraction data through the
product's lineage. The result then goes into the same storage root, as a new product
resource derived from the diffraction data and the input product, and that resource's
folder is the run's output directory: its uuid is printed as the run starts, each checkpoint
becomes its current result as soon as it is written, and it ends ``succeeded``,
``cancelled`` or ``failed`` with whatever the run produced. Without ``--options``, the run
takes the options document the stored product names -- the one a reconstruction ran with,
or the one an imported prepared directory held -- and the stock options when it names none.
A trainable reconstructor has no model to load in this mode.

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
from ptychodus.api.processing import RunRecord, RunState, create_run_record
from ptychodus.api.reconstruct import PositionIndexFilter, ReconstructInput, ReconstructOutput
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    LayoutError,
    ManifestLoadError,
    ResourceKind,
    StorageInputs,
    StorageLayout,
    commit_checkpoint,
    resolve_storage_inputs,
    start_run,
)
from ptychodus.api.options import OptionsError
from ptychodus.api.plugins import PluginRegistry
from ptychodus.cli import (
    CancellationToken,
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    install_signal_handlers,
    open_action_log,
    record_run_end,
    report_resource,
)
from ptychodus.cli._launch import environment_world_size
from ptychodus.cli._reconstruct_common import (
    ResolvedReconstruction,
    add_options_argument,
    add_print_options_argument,
    default_reconstruction,
    find_options_document,
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
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory for -i, written in the ptychodus standard layout: '
            'reconstruct_options.yaml, reconstruct.log, product.NNNNNN.h5 checkpoints, and '
            'the final product.h5. Must differ from INPUT_DIR. Not taken with --product, '
            'whose result is a new product resource in the storage root.'
        ),
    )
    add_options_argument(
        parser,
        default_note=(
            'Defaults to the one in INPUT_DIR, or the one a --product names, else to stock '
            'pty-chi LSQML, or BH for a near-field product.'
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


class _CheckpointCommitter:
    """Publishes each checkpoint a run writes as its product resource's current result."""

    def __init__(self, layout: StorageLayout, uuid: UUID, run: RunRecord) -> None:
        self._layout = layout
        self._uuid = uuid
        self.run = run
        self.last_checkpoint: Path | None = None

    def __call__(self, output: ReconstructOutput, checkpoint_file: Path | None) -> None:
        self.run = self.run.revise(progress=output.progress, progress_goal=output.progress_goal)

        if checkpoint_file is not None:
            commit_checkpoint(self._layout, self._uuid, data_file=checkpoint_file, run=self.run)
            self.last_checkpoint = checkpoint_file


def _reconstruct_into_storage(
    layout: StorageLayout,
    inputs: StorageInputs,
    product_uuid: UUID,
    resolved: ResolvedReconstruction,
    reconstruct_input: ReconstructInput,
    cancellation: CancellationToken,
) -> ExitCode:
    """Reconstruct into a new product resource that is the run's output directory."""
    run = create_run_record(
        RunState.RUNNING,
        command='ptychodus-reconstruct',
        options=StandardFileLayout.RECONSTRUCT_OPTIONS.value,
    )
    uuid, folder = start_run(
        layout,
        ResourceKind.PRODUCT,
        initial=inputs.product_file,
        derived_from=[inputs.diffraction_uuid, product_uuid],
        options_file_name=StandardFileLayout.RECONSTRUCT_OPTIONS.value,
        options_document=resolved.dump(),
        log_file_name=StandardFileLayout.RECONSTRUCT_LOG.value,
        run=run,
    )
    report_resource(logger, ResourceKind.PRODUCT, uuid, folder)
    open_action_log(folder, StandardFileLayout.RECONSTRUCT_LOG)
    committer = _CheckpointCommitter(layout, uuid, run)

    try:
        product = run_reconstruction(
            logger,
            resolved,
            reconstruct_input,
            folder,
            cancellation=cancellation,
            on_sync=committer,
            published_progress=run.progress,
        )
    except Exception as exc:
        logger.exception('Reconstruction failed; product %s keeps its last checkpoint.', uuid)
        record_run_end(
            logger,
            layout,
            uuid,
            data_file=committer.last_checkpoint,
            run=committer.run.revise(state=RunState.FAILED, reason=type(exc).__name__),
        )
        return ExitCode.FAILURE

    # Written by run_reconstruction whenever there was an output to write: at the end of a
    # completed run and when a cancelled one stops. Cancelled before any output, the result
    # stays the input copy the resource started from.
    final_file = None if product is None else StandardFileLayout.PRODUCT.path(folder)

    if cancellation.is_cancelled:
        state, code = RunState.CANCELLED, ExitCode.CANCELLED
    else:
        state, code = RunState.SUCCEEDED, ExitCode.SUCCESS

    if not record_run_end(
        logger, layout, uuid, data_file=final_file, run=committer.run.revise(state=state)
    ):
        return ExitCode.FAILURE

    return code


def main() -> ExitCode:
    parser = _build_parser()
    args = parser.parse_args()

    input_directory: Path | None = args.input_directory
    output_directory: Path | None = args.output_directory
    layout: StorageLayout | None = None
    inputs: StorageInputs | None = None

    if input_directory is None:
        if args.storage_root is None:
            parser.error(f'--product needs --storage-root or ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.')

        if output_directory is not None:
            parser.error('--product writes into a new product resource; drop -o.')

        if environment_world_size() > 1:
            parser.error(
                '--product cannot run under a multi-process launch: its ranks could not agree '
                'on the new resource. Reconstruct a prepared directory with -i and -o instead.'
            )

        layout = StorageLayout(Path(args.storage_root))

        try:
            inputs = resolve_storage_inputs(layout, args.product)
        except (LayoutError, ManifestLoadError) as exc:
            parser.error(str(exc))

        diffraction_file = inputs.diffraction_file
        input_product_file = inputs.product_file
    else:
        if output_directory is None:
            parser.error('-o/--output-directory is required with -i.')

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

    if output_directory is not None and not args.dry_run:
        open_action_log(output_directory, StandardFileLayout.RECONSTRUCT_LOG)

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored.
    cancellation = install_signal_handlers()
    libraries = PluginRegistry.load_plugins().reconstructor_libraries

    # A named or staged document is read before the data are, so options that do not
    # parse fail in seconds rather than after loading several gigabytes. Without one, the
    # stock algorithm follows the propagation regime, which only the product states.
    if input_directory is not None:
        options_file = find_options_document(args.options, input_directory)
    elif args.options is None and inputs is not None and inputs.options_file is not None:
        logger.info('Using the options stored with product %s', args.product)
        options_file = inputs.options_file
    else:
        options_file = args.options

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

        if inputs is None:
            assert output_directory is not None
            logger.info('Would write %s', StandardFileLayout.PRODUCT.path(output_directory))
            logger.info(
                'Would write %s', StandardFileLayout.RECONSTRUCT_OPTIONS.path(output_directory)
            )
        else:
            logger.info(
                'Would create a product resource in %s derived from diffraction %s and product %s.',
                args.storage_root,
                inputs.diffraction_uuid,
                args.product,
            )

        logger.info('Nothing written.')
        return ExitCode.SUCCESS

    if layout is not None and inputs is not None:
        return _reconstruct_into_storage(
            layout, inputs, args.product, resolved, reconstruct_input, cancellation
        )

    assert input_directory is not None and output_directory is not None
    load_trained_model(logger, resolved.reconstructor, input_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_reconstruction(
        logger, resolved, reconstruct_input, output_directory, cancellation=cancellation
    )
    return ExitCode.CANCELLED if cancellation.is_cancelled else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
