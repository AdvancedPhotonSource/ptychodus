#!/usr/bin/env python
"""Enhance a fluorescence dataset against a ptychography product.

The companion to ``ptychodus-reconstruct``: that command turns a prepared directory into a
``product.h5``, and this one hands that product and the fluorescence dataset measured
alongside the same scan to one enhancer, which estimates each element map on the object grid.
It reads no settings file and builds no application model, so it runs anywhere the product
and the dataset can be read.

The enhanced dataset is written in the format the measured one was read in, to a file that
must not be either input. No dataset is written unless the enhancer runs to completion: an
interrupted run leaves no output rather than a dataset with only some maps enhanced. Beside
the output go ``enhance_xrf_options.yaml``, the options the run used as a document
``--options`` takes back, and ``enhance_xrf.log``.

An enhancer's options come from a YAML document naming the enhancer and its fields.
``--print-options`` writes one at its defaults, to edit and pass back with ``--options``.

Given resources in a storage root instead -- ``--product`` and ``--fluorescence`` as uuids --
the enhanced dataset goes into the same root, as a new fluorescence resource derived from
both, and that resource's folder is the run's output directory. It is created before
enhancing starts and its uuid printed, each dataset the enhancer yields becomes its current
result as soon as it is written, and it ends ``succeeded``, ``cancelled`` or ``failed`` with
whatever the run produced, even if that is nothing yet.

SIGINT and SIGTERM ask the enhancer to stop after the step it is on; a second signal ends
the process at once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from uuid import UUID
import argparse
import logging
import os
import sys

from pydantic import BaseModel

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
)
from ptychodus.api.io import (
    StandardFileLayout,
    load_fluorescence_data,
    load_product,
    save_fluorescence_data,
)
from ptychodus.api.plugins import Plugin, PluginCatalog, PluginRegistry
from ptychodus.api.processing import RunState, create_run_record
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    LayoutError,
    ManifestLoadError,
    ResourceKind,
    StorageLayout,
    commit_checkpoint,
    resolve_resource_file,
    start_run,
)
from ptychodus.cli import (
    CancellationToken,
    add_log_level_argument,
    configure_logging,
    install_signal_handlers,
    open_action_log,
    record_run_end,
    report_resource,
)
from ptychodus.api.options import (
    OptionsError,
    apply_options_document,
    dump_options_document,
    read_options_document,
)
from ptychodus.cli._options import format_options_document

logger = logging.getLogger('enhance_xrf')

# The enhancer a run without --algorithm or --options uses, and the one --print-options
# describes when given no name. The same default the application's settings start from.
_DEFAULT_ALGORITHM: Final = 'vspi'
_DEFAULT_FILE_TYPE: Final = 'XRF-Maps'


def format_enhancer_options(plugin: Plugin[FluorescenceEnhancer]) -> str:
    """One enhancer's default options as a commented YAML document.

    The document names the enhancer beside its fields, so it can be saved and passed
    straight back to ``--options``.
    """
    return format_options_document(
        f'{plugin.simple_name} fluorescence enhancer options',
        {'algorithm': plugin.simple_name},
        plugin.strategy.get_default_options(),
    )


def _validate_options(plugin: Plugin[FluorescenceEnhancer], fields: Mapping[str, Any]) -> BaseModel:
    """The enhancer's defaults with `fields` applied, validated against its declaration.

    Raises:
        OptionsError: if a field is not declared, or a value is one the declaration rejects.
    """
    return apply_options_document(
        plugin.strategy.get_default_options(),
        fields,
        subject=f'Enhancer "{plugin.simple_name}"',
    )


class _PrintOptionsAction(argparse.Action):
    """Print one enhancer's default options and exit, during parsing.

    An action rather than a flag ``main`` checks, so that it works without the inputs a run
    needs, and prints before anything else the command would do.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        enhancers: PluginCatalog[FluorescenceEnhancer],
        **kwargs: Any,
    ) -> None:
        self._enhancers = enhancers
        super().__init__(
            option_strings,
            dest,
            nargs='?',
            const=_DEFAULT_ALGORITHM,
            default=argparse.SUPPRESS,
            choices=tuple(plugin.simple_name for plugin in enhancers),
            **kwargs,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        algorithm = _DEFAULT_ALGORITHM if values is None else str(values)
        plugin = self._enhancers.find_plugin(algorithm)

        if plugin is None:
            parser.error(f'No enhancer named "{algorithm}".')

        # Explicitly stdout, so ``--print-options > options.yaml`` captures it;
        # ``parser.exit(message=...)`` would write to stderr.
        parser._print_message(format_enhancer_options(plugin), sys.stdout)
        parser.exit()


def _file_types(registry: PluginRegistry) -> tuple[str, ...]:
    """Names of the formats a dataset can be both read and written in."""
    writer_names = {plugin.simple_name for plugin in registry.fluorescence_file_writers}
    return tuple(
        plugin.simple_name
        for plugin in registry.fluorescence_file_readers
        if plugin.simple_name in writer_names
    )


def _path_or_uuid(text: str) -> Path | UUID:
    """A uuid when `text` parses as one, and otherwise a path."""
    try:
        return UUID(text)
    except ValueError:
        return Path(text)


def _build_parser(registry: PluginRegistry) -> argparse.ArgumentParser:
    enhancers = registry.fluorescence_enhancers
    parser = argparse.ArgumentParser(
        description='Enhance a fluorescence dataset against a ptychography product.',
    )
    parser.add_argument(
        '--product',
        metavar='PRODUCT',
        type=_path_or_uuid,
        help='ptychodus product HDF5 file, such as the product.h5 ptychodus-reconstruct '
        'writes, or the uuid of a product in the storage root. Text that parses as a uuid '
        'is taken as one.',
    )
    parser.add_argument(
        '--fluorescence',
        metavar='FLUORESCENCE',
        type=_path_or_uuid,
        help='Measured fluorescence dataset, in the format --file-type names, or the uuid of '
        'a fluorescence resource in the storage root. Not modified. Both inputs are files or '
        'both are uuids.',
    )
    parser.add_argument(
        '--storage-root',
        metavar='STORAGE_ROOT',
        type=Path,
        default=os.environ.get(STORAGE_ROOT_ENVIRONMENT_VARIABLE),
        help='Storage root holding uuid inputs, where the enhanced dataset is saved as a new '
        f'fluorescence resource. Defaults to ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.',
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        type=Path,
        help='Enhanced dataset, written in the format of --fluorescence. Must differ '
        'from both inputs; an existing file is overwritten. Not taken with uuid inputs.',
    )
    parser.add_argument(
        '--file-type',
        choices=_file_types(registry),
        default=None,
        help='Format the fluorescence dataset is read and the enhanced one written in. '
        f'Defaults to {_DEFAULT_FILE_TYPE}. Not taken with uuid inputs, whose resources are '
        'native HDF5.',
    )
    parser.add_argument(
        '--algorithm',
        choices=tuple(plugin.simple_name for plugin in enhancers),
        default=None,
        help=f'Enhancer to run, at its default options. Defaults to {_DEFAULT_ALGORITHM}, '
        'or to the enhancer an --options file names.',
    )
    parser.add_argument(
        '--options',
        metavar='OPTIONS_FILE',
        type=Path,
        default=None,
        help='YAML naming an enhancer and its options, as --print-options writes; - reads '
        'stdin. Fields it omits keep their defaults.',
    )
    parser.add_argument(
        '--print-options',
        action=_PrintOptionsAction,
        enhancers=enhancers,
        metavar='ALGORITHM',
        help=f"Print this enhancer's default options as YAML, then exit. Defaults to "
        f'{_DEFAULT_ALGORITHM}. Redirect to a file, edit it, and pass it back with --options.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Read and check the inputs and options, and report them, without enhancing.',
    )
    add_log_level_argument(parser)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def _enhance_into_storage(
    layout: StorageLayout,
    plugin: Plugin[FluorescenceEnhancer],
    options: BaseModel,
    input_: FluorescenceEnhancerInput,
    derived_from: Sequence[UUID],
    cancellation: CancellationToken,
) -> ExitCode:
    """Enhance into a new fluorescence resource that is the run's output directory."""
    run = create_run_record(
        RunState.RUNNING,
        command='ptychodus-enhance-xrf',
        options=StandardFileLayout.ENHANCE_XRF_OPTIONS.value,
    )
    uuid, folder = start_run(
        layout,
        ResourceKind.FLUORESCENCE,
        initial=None,
        derived_from=derived_from,
        options_file_name=StandardFileLayout.ENHANCE_XRF_OPTIONS.value,
        options_document=dump_options_document({'algorithm': plugin.simple_name}, options),
        log_file_name=StandardFileLayout.ENHANCE_XRF_LOG.value,
        run=run,
    )
    report_resource(logger, ResourceKind.FLUORESCENCE, uuid, folder)
    open_action_log(folder, StandardFileLayout.ENHANCE_XRF_LOG)

    enhanced: FluorescenceDataset | None = None
    last_checkpoint: Path | None = None

    try:
        for output in plugin.strategy.enhance(input_, options=options):
            enhanced = output.dataset
            run = run.revise(progress=output.progress, progress_goal=output.progress_goal)
            checkpoint = StandardFileLayout.FLUORESCENCE.checkpoint_path(folder, output.progress)

            # A checkpoint is never rewritten; a step that does not advance the progress
            # keeps the one already published for it.
            if not checkpoint.exists():
                save_fluorescence_data(checkpoint, enhanced)
                commit_checkpoint(layout, uuid, data_file=checkpoint, run=run)
                last_checkpoint = checkpoint

            if output.progress_goal > 0:
                logger.info('Enhanced %d/%d', output.progress, output.progress_goal)

            if cancellation.is_cancelled:
                break
    except Exception as exc:
        logger.exception('Enhancement failed; fluorescence %s keeps its last checkpoint.', uuid)
        record_run_end(
            logger,
            layout,
            uuid,
            data_file=last_checkpoint,
            run=run.revise(state=RunState.FAILED, reason=type(exc).__name__),
        )
        return ExitCode.FAILURE

    if cancellation.is_cancelled:
        logger.warning('Enhancement cancelled; fluorescence %s keeps what it reached.', uuid)
        state, code = RunState.CANCELLED, ExitCode.CANCELLED
    elif enhanced is None:
        logger.error('Enhancer %s produced no dataset.', plugin.simple_name)
        record_run_end(
            logger,
            layout,
            uuid,
            data_file=None,
            run=run.revise(state=RunState.FAILED, reason='no-result'),
        )
        return ExitCode.FAILURE
    else:
        state, code = RunState.SUCCEEDED, ExitCode.SUCCESS

    final_file: Path | None = None

    if enhanced is not None:
        final_file = StandardFileLayout.FLUORESCENCE.path(folder)
        save_fluorescence_data(final_file, enhanced)

    if not record_run_end(logger, layout, uuid, data_file=final_file, run=run.revise(state=state)):
        return ExitCode.FAILURE

    return code


def _enhance_to_file(
    plugin: Plugin[FluorescenceEnhancer],
    options: BaseModel,
    input_: FluorescenceEnhancerInput,
    output_file: Path,
    file_type: str,
    registry: PluginRegistry,
    cancellation: CancellationToken,
) -> ExitCode:
    """Enhance into one output file, written only when the enhancer runs to completion."""
    # Beside the output, named for the action: a later run into the same directory
    # replaces the options document and appends to the log.
    output_directory = output_file.parent
    output_directory.mkdir(parents=True, exist_ok=True)
    open_action_log(output_directory, StandardFileLayout.ENHANCE_XRF_LOG)
    options_file = StandardFileLayout.ENHANCE_XRF_OPTIONS.path(output_directory)
    logger.info('Writing %s', options_file)
    options_file.write_text(dump_options_document({'algorithm': plugin.simple_name}, options))

    enhanced: FluorescenceDataset | None = None

    try:
        for output in plugin.strategy.enhance(input_, options=options):
            enhanced = output.dataset

            if output.progress_goal > 0:
                logger.info('Enhanced %d/%d', output.progress, output.progress_goal)

            if cancellation.is_cancelled:
                break
    except Exception:
        logger.exception('Enhancement failed; nothing written.')
        return ExitCode.FAILURE

    if cancellation.is_cancelled:
        logger.warning('Enhancement cancelled; nothing written.')
        return ExitCode.CANCELLED

    if enhanced is None:
        logger.error('Enhancer %s produced no dataset; nothing written.', plugin.simple_name)
        return ExitCode.FAILURE

    writer = registry.fluorescence_file_writers.get_strategy_by_name(file_type)
    writer.write(output_file, enhanced)
    logger.info('Wrote %s', output_file)
    return ExitCode.SUCCESS


def main() -> ExitCode:
    # Before the parser, so its choices name what is actually installed.
    registry = PluginRegistry.load_plugins()
    parser = _build_parser(registry)
    args = parser.parse_args()

    missing = [
        option
        for option, value in (('--product', args.product), ('--fluorescence', args.fluorescence))
        if value is None
    ]

    if missing:
        parser.error(f'the following arguments are required: {", ".join(missing)}')

    product_source: Path | UUID = args.product
    fluorescence_source: Path | UUID = args.fluorescence
    output_file: Path | None = args.output
    layout: StorageLayout | None = None
    file_type = _DEFAULT_FILE_TYPE if args.file_type is None else args.file_type

    if isinstance(product_source, UUID) and isinstance(fluorescence_source, UUID):
        if args.storage_root is None:
            parser.error(
                f'uuid inputs need --storage-root or ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.'
            )

        if output_file is not None:
            parser.error('uuid inputs write into a new fluorescence resource; drop -o.')

        if args.file_type is not None:
            parser.error('uuid inputs are native HDF5 resources; drop --file-type.')

        layout = StorageLayout(Path(args.storage_root))

        try:
            product_file = resolve_resource_file(layout, product_source, ResourceKind.PRODUCT)
            fluorescence_file = resolve_resource_file(
                layout, fluorescence_source, ResourceKind.FLUORESCENCE
            )
        except (LayoutError, ManifestLoadError) as exc:
            parser.error(str(exc))
    elif isinstance(product_source, Path) and isinstance(fluorescence_source, Path):
        if output_file is None:
            parser.error('the following arguments are required: -o/--output')

        product_file = product_source
        fluorescence_file = fluorescence_source

        for input_file in (product_file, fluorescence_file):
            if not input_file.is_file():
                parser.error(f'{input_file} is not a file.')

        if output_file.exists():
            if output_file.is_dir():
                parser.error(f'{output_file} is a directory; name the file to write.')

            for input_file in (product_file, fluorescence_file):
                if output_file.samefile(input_file):
                    parser.error(f'OUTPUT_FILE must differ from the input {input_file}.')
    else:
        parser.error(
            '--product and --fluorescence must both be files or both be uuids: a result '
            'derived from an input outside the storage root could not record its lineage.'
        )

    algorithm: str | None = args.algorithm
    fields: Mapping[str, Any] = {}

    # Before the data are read, so options that cannot be used fail in seconds.
    try:
        if args.options is not None:
            selectors, fields = read_options_document(args.options, ('algorithm',))
            file_algorithm = selectors['algorithm']

            if algorithm is not None and algorithm.casefold() != file_algorithm.casefold():
                raise OptionsError(
                    f'--algorithm {algorithm} disagrees with {args.options}, which names '
                    f'{file_algorithm}; pass one or the other.'
                )

            algorithm = file_algorithm

        plugin = registry.fluorescence_enhancers.find_plugin(algorithm or _DEFAULT_ALGORITHM)

        if plugin is None:
            raise OptionsError(
                f'No enhancer named "{algorithm}"; installed are '
                f'{registry.fluorescence_enhancers.stringify_plugin_names()}.'
            )

        options = _validate_options(plugin, fields)
    except (OptionsError, OSError) as exc:
        parser.error(str(exc))

    configure_logging(args)
    cancellation = install_signal_handlers()

    logger.info('Reading %s', product_file)
    product = load_product(product_file)

    if layout is None:
        logger.info('Reading %s as %s', fluorescence_file, file_type)
        reader = registry.fluorescence_file_readers.get_strategy_by_name(file_type)
        dataset = reader.read(fluorescence_file)
    else:
        logger.info('Reading %s', fluorescence_file)
        dataset = load_fluorescence_data(fluorescence_file)

    logger.info('Enhancer %s, options %s', plugin.simple_name, options.model_dump(mode='json'))

    if args.dry_run:
        logger.info(
            'Dry run: %d element maps of shape %s, %d probe positions, object %s.',
            len(dataset),
            dataset[0].counts_per_second.shape if len(dataset) > 0 else (),
            len(product.probe_positions),
            product.object_.get_array().shape,
        )

        if layout is None:
            logger.info('Would write %s as %s', output_file, file_type)
        else:
            logger.info(
                'Would create a fluorescence resource in %s derived from product %s and '
                'fluorescence %s.',
                args.storage_root,
                product_source,
                fluorescence_source,
            )

        return ExitCode.SUCCESS

    input_ = FluorescenceEnhancerInput(dataset=dataset, product=product)

    if layout is not None:
        assert isinstance(product_source, UUID) and isinstance(fluorescence_source, UUID)
        return _enhance_into_storage(
            layout, plugin, options, input_, [product_source, fluorescence_source], cancellation
        )

    assert output_file is not None

    if output_file.exists():
        logger.warning('%s will be overwritten.', output_file)

    return _enhance_to_file(plugin, options, input_, output_file, file_type, registry, cancellation)


if __name__ == '__main__':
    sys.exit(main())
