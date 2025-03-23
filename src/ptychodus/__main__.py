#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.model import ModelCore
import ptychodus

logger = logging.getLogger(__name__)


def versionString() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def verifyAllArgumentsParsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


def main() -> int:
    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychography analysis application',
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
        version=versionString(),
    )

    parsedArgs, unparsedArgs = parser.parse_known_args()
    settingsFile = Path(parsedArgs.settings.name) if parsedArgs.settings else None

    with ModelCore(settingsFile, is_developer_mode_enabled=parsedArgs.dev) as model:
        if parsedArgs.patterns is not None:
            patternsFilePath = Path(parsedArgs.patterns.name)
            model.workflow_api.import_assembled_patterns(patternsFilePath)

        if parsedArgs.batch is not None:
            verifyAllArgumentsParsed(parser, unparsedArgs)

            if parsedArgs.input is None or parsedArgs.output is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            action = parsedArgs.batch
            inputFilePath = Path(parsedArgs.input.name)
            outputFilePath = Path(parsedArgs.output.name)
            fluorescenceInputFilePath: Path | None = None
            fluorescenceOutputFilePath: Path | None = None

            if parsedArgs.fluorescence_input is not None:
                fluorescenceInputFilePath = Path(parsedArgs.fluorescence_input.name)

            if parsedArgs.fluorescence_output is not None:
                fluorescenceOutputFilePath = Path(parsedArgs.fluorescence_output.name)

            return model.batch_mode_execute(
                action,
                inputFilePath,
                outputFilePath,
                fluorescence_input_file_path=fluorescenceInputFilePath,
                fluorescence_output_file_path=fluorescenceOutputFilePath,
            )

        try:
            from PyQt5.QtWidgets import QApplication
        except ModuleNotFoundError:
            logger.warning('PyQt5 not found.')
            return 0

        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsedArgs)
        verifyAllArgumentsParsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore

        view = ViewCore()

        from ptychodus.controller import ControllerCore

        controller = ControllerCore(model, view)
        controller.show_main_window(versionString())

        return app.exec()


sys.exit(main())
