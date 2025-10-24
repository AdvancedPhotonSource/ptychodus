#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def version_string() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def verify_all_arguments_parsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychography data analysis application',
    )
    parser.add_argument(
        '-b',
        '--batch',
        choices=('reconstruct', 'train'),
        help='Run action non-interactively',
    )
    parser.add_argument(
        '--fluorescence-input',
        metavar='FLUORESCENCE_INPUT_FILE',
        type=argparse.FileType('r'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--fluorescence-output',
        metavar='FLUORESCENCE_OUTPUT_FILE',
        type=argparse.FileType('w'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--batch-input',
        metavar='INPUT_FILE',
        type=argparse.FileType('r'),
        help='Path to the input file (batch mode)',
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.INFO,
        help='Set Python logging level.',
    )
    parser.add_argument(
        '--batch-output',
        metavar='OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='Path to the output file (batch mode)',
    )
    parser.add_argument(
        # preprocessed diffraction patterns file (batch mode)
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        type=argparse.FileType('r'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        help='Path to the settings file.',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=version_string(),
    )

    parsed_args, unparsed_args = parser.parse_known_args()
    settings_file = Path(parsed_args.settings.name) if parsed_args.settings else None

    with ModelCore(settings_file, log_level=parsed_args.log_level) as model:
        if parsed_args.diffraction_input is not None:
            diffraction_file_path = Path(parsed_args.diffraction_input.name)
            model.workflow_api.import_assembled_patterns(diffraction_file_path)

        if parsed_args.batch is not None:
            verify_all_arguments_parsed(parser, unparsed_args)

            if parsed_args.batch_input is None or parsed_args.batch_output is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            action = parsed_args.batch
            input_file_path = Path(parsed_args.batch_input.name)
            output_file_path = Path(parsed_args.batch_output.name)
            fluorescence_input_file_path: Path | None = None
            fluorescence_output_file_path: Path | None = None

            if parsed_args.fluorescence_input is not None:
                fluorescence_input_file_path = Path(parsed_args.fluorescence_input.name)

            if parsed_args.fluorescence_output is not None:
                fluorescence_output_file_path = Path(parsed_args.fluorescence_output.name)

            return model.batch_mode_execute(
                action,
                input_file_path,
                output_file_path,
                fluorescence_input_file_path=fluorescence_input_file_path,
                fluorescence_output_file_path=fluorescence_output_file_path,
            )

        try:
            from PyQt5.QtWidgets import QApplication
        except ModuleNotFoundError:
            logger.warning('PyQt5 not found.')
            return 0

        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsed_args)
        verify_all_arguments_parsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore

        view = ViewCore()

        from ptychodus.controller import ControllerCore

        controller = ControllerCore(
            model, view, is_developer_mode_enabled=model.is_developer_mode_enabled
        )
        controller.show_main_window(version_string())

        return app.exec()


sys.exit(main())
