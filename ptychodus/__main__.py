#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

from ptychodus.model import ModelArgs, ModelCore
import ptychodus

import h5py
import matplotlib
import numpy


def versionString() -> str:
    return f'{ptychodus.__name__.title()} ({ptychodus.__version__})'


def verifyAllArgumentsParsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


def main() -> int:
    NOT_FOUND_STR = 'NOT FOUND!'

    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychography analysis front-end')
    parser.add_argument('-b', '--batch', action='store_true', \
            help='run reconstruction non-interactively')
    parser.add_argument('-d', '--dev', action='store_true', help='run in developer mode')
    parser.add_argument('-f', '--file-prefix', action='store', dest='prefix', \
            help='replace file path prefix')
    parser.add_argument('-p', '--port', action='store', type=int, default=9999, \
            help='remote process communication port number')
    parser.add_argument('-s', '--settings', action='store', type=argparse.FileType('r'), \
            help='use settings from file')
    parser.add_argument('-v', '--version', action='version', version=versionString())
    parsedArgs, unparsedArgs = parser.parse_known_args()

    logger = logging.getLogger(ptychodus.__name__)
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        stream=sys.stdout,
                        encoding='utf-8',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info(versionString())
    logger.info(f'\tNumPy {numpy.__version__}')
    logger.info(f'\tMatplotlib {matplotlib.__version__}')
    logger.info(f'\tHDF5 {h5py.version.hdf5_version}')
    logger.info(f'\tH5Py {h5py.__version__}')

    modelArgs = ModelArgs(
        rpcPort=parsedArgs.port,
        autoExecuteRPCs=False,  # TODO False if using GUI else True
        replacementPathPrefix=parsedArgs.prefix,
        isDeveloperModeEnabled=parsedArgs.dev)

    with ModelCore(modelArgs) as model:
        if parsedArgs.settings:
            model.settingsRegistry.openSettings(parsedArgs.settings.name)

        if parsedArgs.batch:
            verifyAllArgumentsParsed(parser, unparsedArgs)
            return model.batchModeReconstruct()

        try:
            import PyQt5
            from PyQt5.QtWidgets import QApplication
            from PyQt5.Qt import PYQT_VERSION_STR
            from PyQt5.QtCore import QT_VERSION_STR
        except ModuleNotFoundError:
            logger.error('\tPyQt ' + NOT_FOUND_STR)
            return -1

        logger.info('\tQt ' + QT_VERSION_STR if QT_VERSION_STR else NOT_FOUND_STR)
        logger.info('\tPyQt ' + PYQT_VERSION_STR if PYQT_VERSION_STR else NOT_FOUND_STR)

        # QApplication expects the first argument to be the program name
        qtArgs = sys.argv[:1] + unparsedArgs
        app = QApplication(qtArgs)
        verifyAllArgumentsParsed(parser, app.arguments()[1:])

        from ptychodus.view import ViewCore
        view = ViewCore.createInstance()

        from ptychodus.controller import ControllerCore
        controller = ControllerCore.createInstance(model, view)

        view.setWindowTitle(versionString())
        view.show()
        return app.exec_()


sys.exit(main())
