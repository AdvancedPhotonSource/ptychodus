#!/usr/bin/env python

from pathlib import Path
from typing import Optional
import argparse
import logging
import sys

from ptychodus.model import ModelArgs, ModelCore
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
        action='store',
        type=argparse.FileType('w'),
        metavar='RESULTS_FILE',
        help='run reconstruction non-interactively',
    )
    parser.add_argument(
        '-d',
        '--dev',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-f',
        '--file-prefix',
        action='store',
        dest='prefix',
        help='replace file path prefix in settings',
    )
    parser.add_argument(
        '-p',
        '--port',
        action='store',
        default=9999,
        help='remote process communication port number',
        type=int,
    )
    parser.add_argument(
        '-r',
        '--restart',
        action='store',
        help='use restart data from file',
        metavar='RESTART_FILE',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        '-s',
        '--settings',
        action='store',
        help='use settings from file',
        metavar='SETTINGS_FILE',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=versionString(),
    )
    parsedArgs, unparsedArgs = parser.parse_known_args()

    modelArgs = ModelArgs(
        restartFilePath=Path(parsedArgs.restart.name) if parsedArgs.restart else None,
        settingsFilePath=Path(parsedArgs.settings.name) if parsedArgs.settings else None,
        replacementPathPrefix=parsedArgs.prefix,
        rpcPort=parsedArgs.port,
        autoExecuteRPCs=bool(parsedArgs.batch),
        isDeveloperModeEnabled=parsedArgs.dev,
    )

    with ModelCore(modelArgs) as model:
        if parsedArgs.batch is not None:
            resultsFilePath = Path(parsedArgs.batch.name)
            verifyAllArgumentsParsed(parser, unparsedArgs)
            return model.batchModeReconstruct(resultsFilePath)

        from PyQt5.QtWidgets import QApplication
        # QApplication expects the first argument to be the program name
        app = QApplication(sys.argv[:1] + unparsedArgs)
        verifyAllArgumentsParsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore
        view = ViewCore.createInstance(parsedArgs.dev)

        from ptychodus.controller import ControllerCore
        controller = ControllerCore.createInstance(model, view)

        view.setWindowTitle(versionString())
        view.show()
        return app.exec_()


sys.exit(main())
