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
        help='run action non-interactively',
    )
    parser.add_argument(
        '-d',
        '--dev',
        action='store_true',
        help=argparse.SUPPRESS,
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
        '-i',
        '--input',
        metavar='INPUT_FILE',
        type=argparse.FileType('r'),
        help='input file (batch mode)',
    )
    parser.add_argument(
        '-o',
        '--output',
        metavar='OUTPUT_FILE',
        type=argparse.FileType('w'),
        help='output file (batch mode)',
    )
    parser.add_argument(
        # preprocessed diffraction patterns file (batch mode)
        '-p',
        '--patterns',
        metavar='PATTERNS_FILE',
        type=argparse.FileType('r'),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        help='use settings from file',
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

    with ModelCore(settings_file, is_developer_mode_enabled=parsed_args.dev) as model:
        if parsed_args.patterns is not None:
            patterns_file_path = Path(parsed_args.patterns.name)
            model.workflow_api.import_assembled_patterns(patterns_file_path)

        if parsed_args.batch is not None:
            verify_all_arguments_parsed(parser, unparsed_args)

            if parsed_args.input is None or parsed_args.output is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            action = parsed_args.batch
            input_file_path = Path(parsed_args.input.name)
            output_file_path = Path(parsed_args.output.name)
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

        controller = ControllerCore(model, view, is_developer_mode_enabled=parsed_args.dev)
        controller.show_main_window(version_string())

        return app.exec()


sys.exit(main())
