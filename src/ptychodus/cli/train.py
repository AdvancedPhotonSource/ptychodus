#!/usr/bin/env python
"""Train a reconstructor's model from a directory of training data.

The training data are in the backend's own format -- for PtychoPINN, every ``*.npz``
archive in the directory -- as the GUI or ``ptychodus-generate-training-data`` writes them,
and are only read. The backend writes what it trains into the output
directory, and the main process adds the trained model there as ``model`` with the backend's
extension, the name a later run reads it under, beside ``train_options.yaml`` (the options
it trained with, as a document ``--options`` takes back) and ``train.log``. It reads no
settings file and builds no application model.

A backend's options come from a YAML document naming the library, the algorithm and their
fields. ``--print-options`` writes one at its defaults, to edit and pass back with
``--options``.

**Under torchrun.** ``torchrun --standalone --nproc-per-node=N -m ptychodus.cli.train ...``
runs this command once per rank, and each runs the backend in its own process, joining one
process group from the launcher's variables. Every rank
makes the same checks in the same order, so a run that cannot go ahead stops on every rank;
only the main process logs progress and saves the model. A backend that cannot train one model
across ranks is refused rather than run once per rank into the same directory, and so is a
launch across more than one node. A SIGTERM -- which torchrun sends to the surviving ranks
when one fails -- unwinds the backend as an interrupt.

No GPU selection side effects: choose devices with ``CUDA_VISIBLE_DEVICES``. Under torchrun
every rank on a node must see the same devices, as many as ``--nproc-per-node``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import argparse
import logging
import signal
import sys

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import Plugin, PluginCatalog, PluginRegistry
from ptychodus.api.reconstruct import (
    ReconstructorLibrary,
    TrainableReconstructor,
    TrainOutput,
)
from ptychodus.cli import DirectoryType, add_log_level_argument, configure_logging, open_action_log
from ptychodus.cli._launch import (
    environment_node_count,
    environment_rank,
    environment_world_size,
)
from ptychodus.api.options import (
    OptionsError,
    apply_options_document,
    dump_options_document,
    read_options_document,
)
from ptychodus.cli._options import format_options_document, resolve_trainer

logger = logging.getLogger('train')


def format_training_options(
    plugin: Plugin[ReconstructorLibrary], trainer: TrainableReconstructor
) -> str:
    """One algorithm's default training options as a commented YAML document."""
    return format_options_document(
        f'{plugin.display_name} {trainer.name} training options',
        {'library': plugin.simple_name, 'algorithm': trainer.name},
        trainer.get_default_options(),
    )


class _PrintOptionsAction(argparse.Action):
    """Print one algorithm's default options and exit, during parsing.

    An action rather than a flag ``main`` checks, so that it works without the inputs a run
    needs, and prints before anything else the command would do.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        libraries: PluginCatalog[ReconstructorLibrary],
        **kwargs: Any,
    ) -> None:
        self._libraries = libraries
        super().__init__(option_strings, dest, default=argparse.SUPPRESS, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        library_name, _, algorithm_name = str(values).partition('/')

        try:
            plugin, trainer = resolve_trainer(self._libraries, library_name, algorithm_name or None)
        except OptionsError as exc:
            parser.error(str(exc))

        # Explicitly stdout, so ``--print-options ... > options.yaml`` captures it;
        # ``parser.exit(message=...)`` would write to stderr.
        parser._print_message(format_training_options(plugin, trainer), sys.stdout)
        parser.exit()


def _build_parser(registry: PluginRegistry) -> argparse.ArgumentParser:
    libraries = registry.reconstructor_libraries
    parser = argparse.ArgumentParser(
        description="Train a reconstructor's model from a directory of training data.",
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=True),
        help="Training data in the backend's own format, as the GUI exports it. Not modified.",
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help='Where the backend writes what it trains, and the main process the trained model '
        'as model<ext>. Must differ from INPUT_DIR.',
    )
    parser.add_argument(
        '--library',
        help=f'Reconstructor library to train, one of {libraries.stringify_plugin_names()}. '
        'Defaults to the library an --options file names.',
    )
    parser.add_argument(
        '--algorithm',
        help='Algorithm within the library. Defaults to the one an --options file names, or to '
        'the only one the library trains.',
    )
    parser.add_argument(
        '--options',
        metavar='OPTIONS_FILE',
        type=Path,
        default=None,
        help='YAML naming a library, an algorithm and their options, as --print-options '
        'writes; - reads stdin. Fields it omits keep their defaults.',
    )
    parser.add_argument(
        '--print-options',
        action=_PrintOptionsAction,
        libraries=libraries,
        metavar='LIBRARY[/ALGORITHM]',
        help="Print an algorithm's default options as YAML, then exit. Redirect to a file, "
        'edit it, and pass it back with --options.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check the inputs, options and launch, and report them, without training.',
    )
    add_log_level_argument(parser)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def _agreed(flag: str, flag_value: str | None, document_value: str | None) -> str | None:
    """The name `flag` and the options document agree on, or whichever of them gave one.

    Raises:
        OptionsError: if both gave a name and the names differ.
    """
    if flag_value is None:
        return document_value

    if document_value is not None and flag_value.casefold() != document_value.casefold():
        raise OptionsError(
            f'{flag} {flag_value} disagrees with the options file, which names '
            f'{document_value}; pass one or the other.'
        )

    return flag_value


def _log_progress(output: TrainOutput) -> None:
    losses = [
        f'{label} loss {values[-1].value:.6g}'
        for label, values in (
            ('training', output.training_loss),
            ('validation', output.validation_loss),
        )
        if values
    ]
    progress = (
        f'{output.progress}/{output.progress_goal}'
        if output.progress_goal > 0
        else str(output.progress)
    )
    logger.info('Trained %s%s', progress, ''.join(f', {loss}' for loss in losses))


def main() -> ExitCode:
    # Before the parser, so its help names what is actually installed.
    registry = PluginRegistry.load_plugins()
    parser = _build_parser(registry)
    args = parser.parse_args()

    input_directory: Path = args.input_directory
    output_directory: Path = args.output_directory

    if output_directory.resolve() == input_directory.resolve():
        parser.error('OUTPUT_DIR must differ from INPUT_DIR.')

    # Before the backend is asked anything, so options that cannot be used fail in seconds.
    try:
        library_name: str | None = args.library
        algorithm_name: str | None = args.algorithm
        fields: dict[str, Any] = {}

        if args.options is not None:
            selectors, document_fields = read_options_document(
                args.options, ('library', 'algorithm')
            )
            fields = dict(document_fields)
            library_name = _agreed('--library', library_name, selectors['library'])
            algorithm_name = _agreed('--algorithm', algorithm_name, selectors['algorithm'])

        if library_name is None:
            raise OptionsError('Name a library with --library or an --options file.')

        plugin, trainer = resolve_trainer(
            registry.reconstructor_libraries, library_name, algorithm_name
        )
        subject = f'{plugin.display_name} {trainer.name}'
        options = apply_options_document(trainer.get_default_options(), fields, subject=subject)
    except (OptionsError, OSError) as exc:
        parser.error(str(exc))

    # Read from the environment alone, as each rank was launched, so every rank reaches the
    # same verdict and a refused launch stops them all.
    rank = environment_rank()
    world_size = environment_world_size()
    node_count = environment_node_count()

    if node_count > 1:
        parser.error(f'Training across {node_count} nodes is not supported; launch on one node.')

    if world_size > 1 and not trainer.supports_distributed_training():
        parser.error(
            f'{subject} cannot train one model across {world_size} processes; '
            'launch a single process.'
        )

    configure_logging(args)
    is_main_process = rank == 0
    model_file = output_directory / (
        f'{StandardFileLayout.MODEL_BASENAME}{trainer.get_model_file_extension()}'
    )

    if is_main_process:
        logger.info(
            'Training %s from %s into %s, %d process(es)',
            subject,
            input_directory,
            output_directory,
            world_size,
        )
        logger.info('Options %s', options.model_dump(mode='json'))

    if args.dry_run:
        if is_main_process:
            logger.info('Dry run: would write the trained model to %s', model_file)

        return ExitCode.SUCCESS

    open_action_log(output_directory, StandardFileLayout.TRAIN_LOG, rank=rank)

    if is_main_process:
        options_file = StandardFileLayout.TRAIN_OPTIONS.path(output_directory)
        logger.info('Writing %s', options_file)
        options_file.write_text(
            dump_options_document(
                {'library': plugin.simple_name, 'algorithm': trainer.name}, options
            )
        )

    # A scheduler or launcher ends a job with SIGTERM. Raised as an interrupt it unwinds
    # through the backend, so whatever it holds open is released rather than abandoned.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    trained_model: Path | None = None

    # In this process: a one-shot command gains nothing from a child of its own, and a
    # distributed launcher's ranks are then exactly the processes it started.
    try:
        for output in trainer.train(input_directory, output_directory, options=options):
            if output.trained_model is not None:
                trained_model = output.trained_model

            if is_main_process:
                _log_progress(output)
    except KeyboardInterrupt:
        logger.warning('Training cancelled.')
        return ExitCode.CANCELLED
    except Exception:
        logger.exception('Training failed.')
        return ExitCode.FAILURE

    if not is_main_process:
        return ExitCode.SUCCESS

    if trained_model is None:
        logger.error('%s finished without reporting a trained model; none saved.', subject)
        return ExitCode.FAILURE

    try:
        trainer.load_model(trained_model)
        trainer.save_model(model_file)
    except Exception:
        logger.exception('Saving the trained model to %s failed.', model_file)
        return ExitCode.FAILURE

    logger.info('Wrote %s', model_file)
    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
