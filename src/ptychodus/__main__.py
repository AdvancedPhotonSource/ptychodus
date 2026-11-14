#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.api.exit_codes import ExitCode
from ptychodus.cli import verify_all_arguments_parsed
from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=(
            f'{ptychodus.__name__} is a ptychography data pipeline application. This command '
            'starts the GUI; the ptychodus-* commands run headless.'
        ),
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.INFO,
        help='Set Python logging level.',
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

    try:
        from PyQt5.QtWidgets import QApplication
    except ModuleNotFoundError:
        logging.basicConfig(level=parsed_args.log_level, stream=sys.stderr)
        logger.error(
            'The GUI needs PyQt5, from the "gui" extra. Headless work runs through the '
            'ptychodus-* commands, such as ptychodus-reconstruct.'
        )
        return ExitCode.FAILURE

    with ModelCore(settings_file, log_level=parsed_args.log_level) as model:
        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsed_args)
        verify_all_arguments_parsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore

        view = ViewCore(is_developer_mode_enabled=model.is_developer_mode_enabled)

        from ptychodus.controller.core import ControllerCore

        controller = ControllerCore(
            model, view, is_developer_mode_enabled=model.is_developer_mode_enabled
        )
        controller.show_main_window(ptychodus.VERSION_STRING)

        return app.exec()


if __name__ == '__main__':
    sys.exit(main())
