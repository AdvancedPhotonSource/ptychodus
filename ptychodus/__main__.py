#!/usr/bin/env python

from pathlib import Path
import argparse
import logging
import sys

import ptychodus.model
import ptychodus.view
import ptychodus.controller

import h5py
import matplotlib
import numpy

try:
    import PyQt5
    from PyQt5.QtWidgets import QApplication
    from PyQt5.Qt import PYQT_VERSION_STR
    from PyQt5.QtCore import QT_VERSION_STR
except ImportError:
    PyQt5 = None


def versionString() -> str:
    return f'{ptychodus.__name__} ({ptychodus.__version__})'


def verifyAllArgumentsParsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


def main() -> int:
    NOT_FOUND_STR = 'NOT FOUND!'

    parser = argparse.ArgumentParser(
        prog=ptychodus.__name__.lower(),
        description=f'{ptychodus.__name__} is a ptychographic reconstruction user interface')
    parser.add_argument('-b',
                        '--batch',
                        action='store_true',
                        help='run reconstruction non-interactively')
    parser.add_argument('-d', '--dev', action='store_true', help='run in developer mode')
    parser.add_argument('-s',
                        '--settings',
                        action='store',
                        type=argparse.FileType('r'),
                        help='use settings from file')
    parser.add_argument('-v', '--version', action='version', version=versionString())
    parsedArgs, unparsedArgs = parser.parse_known_args()

    logger = logging.getLogger(ptychodus.__name__)
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.debug(versionString())
    logger.debug(f'\tNumPy {numpy.__version__}')
    logger.debug(f'\tMatplotlib {matplotlib.__version__}')
    logger.debug(f'\tHDF5 {h5py.version.hdf5_version}')
    logger.debug(f'\tH5Py {h5py.__version__}')
    logger.debug('\tQt ' + QT_VERSION_STR if QT_VERSION_STR else NOT_FOUND_STR)
    logger.debug('\tPyQt ' + PYQT_VERSION_STR if PYQT_VERSION_STR else NOT_FOUND_STR)

    result = 0

    with ptychodus.model.ModelCore.createInstance(isDeveloperModeEnabled=parsedArgs.dev) as model:
        if parsedArgs.settings:
            model.settingsRegistry.read(parsedArgs.settings.name)

        if parsedArgs.batch:
            verifyAllArgumentsParsed(parser, unparsedArgs)
            result = model.reconstructorPresenter.reconstruct()
        elif PyQt5:
            # QApplication expects the first argument to be the program name
            qtArgs = sys.argv[:1] + unparsedArgs
            app = QApplication(qtArgs)
            verifyAllArgumentsParsed(parser, app.arguments()[1:])

            view = ptychodus.view.ViewCore.createInstance()
            controller = ptychodus.controller.ControllerCore.createInstance(model, view)

            view.setWindowTitle(versionString())
            view.show()
            result = app.exec_()
        else:
            logger.error('PyQt5 ' + NOT_FOUND_STR)
            result = -1

    return result


sys.exit(main())
