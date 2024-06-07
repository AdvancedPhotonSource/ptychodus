#!/usr/bin/env python

from pathlib import Path
import argparse
import sys

from ptychodus.model import ModelCore
import ptychodus


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

    with ModelCore(settingsFile, isDeveloperModeEnabled=parsedArgs.dev) as model:
        if parsedArgs.patterns is not None:
            patternsFilePath = Path(parsedArgs.patterns.name)
            model.workflowAPI.importProcessedPatterns(patternsFilePath)

        if parsedArgs.batch is not None:
            verifyAllArgumentsParsed(parser, unparsedArgs)

            if parsedArgs.input is None or parsedArgs.output is None:
                parser.error('Batch mode requires input and output arguments!')
                return -1

            action = parsedArgs.batch
            inputFilePath = Path(parsedArgs.input.name)
            outputFilePath = Path(parsedArgs.output.name)
            return model.batchModeExecute(action, inputFilePath, outputFilePath)

        try:
            from PyQt5.QtWidgets import QApplication
        except ModuleNotFoundError:
            return 0

        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsedArgs)
        verifyAllArgumentsParsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore
        view = ViewCore.createInstance(parsedArgs.dev)

        from ptychodus.controller import ControllerCore
        controller = ControllerCore.createInstance(model, view)
        controller.showMainWindow(versionString())

        return app.exec()


sys.exit(main())
