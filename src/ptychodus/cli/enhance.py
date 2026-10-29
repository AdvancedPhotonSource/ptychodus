#!/usr/bin/env python
"""Enhance a fluorescence dataset against a ptychography product.

The companion to ``ptychodus-reconstruct``: that command turns a prepared directory into a
``product.h5``, and this one hands that product and the fluorescence dataset measured
alongside the same scan to one enhancer, which estimates each element map on the object grid.
It reads no settings file and builds no application model, so it runs anywhere the product
and the dataset can be read.

The enhanced dataset is written in the format the measured one was read in, to a file that
must not be either input. Nothing is written unless the enhancer runs to completion: an
interrupted run leaves no output rather than a dataset with only some maps enhanced.

An enhancer's options come from a YAML document naming the enhancer and its fields.
``--print-options`` writes one at its defaults, to edit and pass back with ``--options``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
import argparse
import logging
import signal
import sys

from pydantic import BaseModel, ValidationError
import yaml

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
)
from ptychodus.api.io import load_product
from ptychodus.api.plugins import Plugin, PluginChooser, PluginRegistry
from ptychodus.cli import add_log_level_argument, configure_logging

logger = logging.getLogger('enhance')

# The enhancer a run without --algorithm or --options uses, and the one --print-options
# describes when given no name. The same default the application's settings start from.
_DEFAULT_ALGORITHM: Final = 'vspi'
_DEFAULT_FILE_TYPE: Final = 'XRF-Maps'

# The conventional "read it from stdin instead" filename, as argparse hands it over once a
# bare ``-`` has been through ``type=Path``.
_STDIN_ARGUMENT: Final = Path('-')


class _OptionsError(ValueError):
    """An options document that cannot be used, reported as a usage error."""


def format_enhancer_options(plugin: Plugin[FluorescenceEnhancer]) -> str:
    """One enhancer's default options as a commented YAML document.

    The document names the enhancer beside its fields, so it can be saved and passed
    straight back to ``--options``. Each field's description is written above it as a
    YAML comment, which :func:`yaml.safe_load` discards.
    """
    defaults = plugin.strategy.get_default_options()
    values = defaults.model_dump(mode='json')
    lines = [
        f'# {plugin.simple_name} fluorescence enhancer options, at their defaults. Save this,',
        '# edit it, and pass it back with --options.',
        yaml.safe_dump({'algorithm': plugin.simple_name}, sort_keys=False).rstrip(),
        'options:',
    ]

    for name, field in type(defaults).model_fields.items():
        if field.description:
            lines.append(f'  # {field.description}')

        entry = yaml.safe_dump({name: values[name]}, default_flow_style=False, sort_keys=False)
        lines.extend(f'  {line}' for line in entry.splitlines())

    return '\n'.join(lines) + '\n'


def _read_options_document(options_file: Path) -> tuple[str, Mapping[str, Any]]:
    """The enhancer an options document names, and the field values it gives.

    Raises:
        _OptionsError: if the document is not a mapping holding a string ``algorithm``
            and, optionally, a mapping of ``options``.
    """
    text = sys.stdin.read() if options_file == _STDIN_ARGUMENT else options_file.read_text()

    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _OptionsError(f'{options_file} is not YAML: {exc}') from exc

    if not isinstance(document, Mapping):
        raise _OptionsError(f'{options_file} must hold a mapping with "algorithm" and "options".')

    unknown_keys = sorted(str(key) for key in document if key not in ('algorithm', 'options'))

    if unknown_keys:
        raise _OptionsError(
            f'{options_file} has unexpected top-level keys {unknown_keys}; '
            'only "algorithm" and "options" are read.'
        )

    algorithm = document.get('algorithm')

    if not isinstance(algorithm, str):
        raise _OptionsError(f'{options_file} must name its enhancer as a string "algorithm".')

    fields = document.get('options')

    if fields is None:
        fields = {}
    elif not isinstance(fields, Mapping):
        raise _OptionsError(f'"options" in {options_file} must be a mapping of field values.')

    return algorithm, fields


def _validate_options(plugin: Plugin[FluorescenceEnhancer], fields: Mapping[str, Any]) -> BaseModel:
    """The enhancer's defaults with `fields` applied, validated against its declaration.

    Raises:
        _OptionsError: if a field is not declared, or a value is one the declaration rejects.
    """
    defaults = plugin.strategy.get_default_options()
    declared = type(defaults).model_fields
    # Checked here because pydantic ignores names a model does not declare, which would
    # let a misspelled field run silently at its default.
    unknown = sorted(str(name) for name in fields if name not in declared)

    if unknown:
        raise _OptionsError(
            f'Enhancer "{plugin.simple_name}" does not declare {unknown}; '
            f'it declares {sorted(declared)}.'
        )

    try:
        return type(defaults).model_validate({**defaults.model_dump(), **fields})
    except ValidationError as exc:
        raise _OptionsError(f'Invalid options for enhancer "{plugin.simple_name}": {exc}') from exc


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
        enhancers: PluginChooser[FluorescenceEnhancer],
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


def _build_parser(registry: PluginRegistry) -> argparse.ArgumentParser:
    enhancers = registry.fluorescence_enhancers
    parser = argparse.ArgumentParser(
        description='Enhance a fluorescence dataset against a ptychography product.',
    )
    parser.add_argument(
        '--product',
        metavar='PRODUCT_FILE',
        type=Path,
        help='ptychodus product HDF5 file, such as the product.h5 ptychodus-reconstruct writes.',
    )
    parser.add_argument(
        '--fluorescence',
        metavar='FLUORESCENCE_FILE',
        type=Path,
        help='Measured fluorescence dataset, in the format --file-type names. Not modified.',
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        type=Path,
        help='Enhanced dataset, written in the same format as FLUORESCENCE_FILE. Must differ '
        'from both inputs; an existing file is overwritten.',
    )
    parser.add_argument(
        '--file-type',
        choices=_file_types(registry),
        default=_DEFAULT_FILE_TYPE,
        help='Format the fluorescence dataset is read and the enhanced one written in. '
        f'Defaults to {_DEFAULT_FILE_TYPE}.',
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


def main() -> ExitCode:
    # Before the parser, so its choices name what is actually installed.
    registry = PluginRegistry.load_plugins()
    parser = _build_parser(registry)
    args = parser.parse_args()

    missing = [
        option
        for option, value in (
            ('--product', args.product),
            ('--fluorescence', args.fluorescence),
            ('-o/--output', args.output),
        )
        if value is None
    ]

    if missing:
        parser.error(f'the following arguments are required: {", ".join(missing)}')

    product_file: Path = args.product
    fluorescence_file: Path = args.fluorescence
    output_file: Path = args.output

    for input_file in (product_file, fluorescence_file):
        if not input_file.is_file():
            parser.error(f'{input_file} is not a file.')

    if output_file.exists():
        if output_file.is_dir():
            parser.error(f'{output_file} is a directory; name the file to write.')

        for input_file in (product_file, fluorescence_file):
            if output_file.samefile(input_file):
                parser.error(f'OUTPUT_FILE must differ from the input {input_file}.')

    algorithm: str | None = args.algorithm
    fields: Mapping[str, Any] = {}

    # Before the data are read, so options that cannot be used fail in seconds.
    try:
        if args.options is not None:
            file_algorithm, fields = _read_options_document(args.options)

            if algorithm is not None and algorithm.casefold() != file_algorithm.casefold():
                raise _OptionsError(
                    f'--algorithm {algorithm} disagrees with {args.options}, which names '
                    f'{file_algorithm}; pass one or the other.'
                )

            algorithm = file_algorithm

        plugin = registry.fluorescence_enhancers.find_plugin(algorithm or _DEFAULT_ALGORITHM)

        if plugin is None:
            raise _OptionsError(
                f'No enhancer named "{algorithm}"; installed are '
                f'{registry.fluorescence_enhancers.stringify_plugin_names()}.'
            )

        options = _validate_options(plugin, fields)
    except (_OptionsError, OSError) as exc:
        parser.error(str(exc))

    configure_logging(args)

    reader = registry.fluorescence_file_readers.get_strategy_by_name(args.file_type)
    writer = registry.fluorescence_file_writers.get_strategy_by_name(args.file_type)

    logger.info('Reading %s', product_file)
    product = load_product(product_file)
    logger.info('Reading %s as %s', fluorescence_file, args.file_type)
    dataset = reader.read(fluorescence_file)
    logger.info('Enhancer %s, options %s', plugin.simple_name, options.model_dump(mode='json'))

    if args.dry_run:
        logger.info(
            'Dry run: %d element maps of shape %s, %d probe positions, object %s.',
            len(dataset),
            dataset[0].counts_per_second.shape if len(dataset) > 0 else (),
            len(product.probe_positions),
            product.object_.get_array().shape,
        )
        logger.info('Would write %s as %s', output_file, args.file_type)
        return ExitCode.SUCCESS

    if output_file.exists():
        logger.warning('%s will be overwritten.', output_file)

    # A scheduler ends a job with SIGTERM. Raised as an interrupt it unwinds through the
    # enhancer, so a backend that spawned a child tears it down rather than orphaning it.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    enhanced: FluorescenceDataset | None = None
    input_ = FluorescenceEnhancerInput(dataset=dataset, product=product)

    try:
        for output in plugin.strategy.enhance(input_, options=options):
            enhanced = output.dataset

            if output.progress_goal > 0:
                logger.info('Enhanced %d/%d', output.progress, output.progress_goal)
    except KeyboardInterrupt:
        logger.warning('Enhancement cancelled; nothing written.')
        return ExitCode.CANCELLED
    except Exception:
        logger.exception('Enhancement failed; nothing written.')
        return ExitCode.FAILURE

    if enhanced is None:
        logger.error('Enhancer %s produced no dataset; nothing written.', plugin.simple_name)
        return ExitCode.FAILURE

    output_file.parent.mkdir(parents=True, exist_ok=True)
    writer.write(output_file, enhanced)
    logger.info('Wrote %s', output_file)
    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
