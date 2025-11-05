#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.cli import DirectoryType, verify_all_arguments_parsed
from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychography data pipeline application',
    )
    parser.add_argument(
        '-b',
        '--batch',
        choices=('reconstruct', 'train'),
        help='Run action non-interactively',
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        type=DirectoryType(must_exist=True),
        help='Path to the input data directory (batch mode only)',
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.INFO,
        help='Set Python logging level.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        type=DirectoryType(must_exist=False),
        help='Path to the output data directory (batch mode only)',
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
        version=ptychodus.VERSION_STRING,
    )

    parsed_args, unparsed_args = parser.parse_known_args()
    settings_file = Path(parsed_args.settings.name) if parsed_args.settings else None

    with ModelCore(settings_file, log_level=parsed_args.log_level) as model:
        if parsed_args.batch is not None:
            verify_all_arguments_parsed(parser, unparsed_args)

            if parsed_args.input_directory is None or parsed_args.output_directory is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            return model.batch_mode_execute(
                parsed_args.batch, parsed_args.input_directory, parsed_args.output_directory
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
        controller.show_main_window(ptychodus.VERSION_STRING)

        return app.exec()


sys.exit(main())
