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


def version_string() -> str:
    return f'{ptychodus.__name__} ({ptychodus.__version__})'


def main() -> int:
    NOT_FOUND_STR = 'NOT FOUND!'

    parser = argparse.ArgumentParser(prog=ptychodus.__name__.lower(),
            description=f'{ptychodus.__name__} is a ptychographic reconstruction user interface')
    parser.add_argument('-b', '--batch', action='store_true',
            help='run reconstruction non-interactively')
    parser.add_argument('-d', '--dev', action='store_true',
            help='run in developer mode')
    parser.add_argument('-s', '--settings', action='store', type=argparse.FileType('r'),
            help='use settings from file')
    parser.add_argument('-v', '--version', action='version', version=version_string())
    parsed_args, unparsed_args = parser.parse_known_args()

    logger = logging.getLogger(ptychodus.__name__)
    logger.setLevel(logging.DEBUG)

    logger.debug(version_string())
    logger.debug(f'\tNumPy {numpy.__version__}')
    logger.debug(f'\tMatplotlib {matplotlib.__version__}')
    logger.debug(f'\tHDF5 {h5py.version.hdf5_version}')
    logger.debug(f'\tH5Py {h5py.__version__}')
    logger.debug('\tQt ' + QT_VERSION_STR if QT_VERSION_STR else NOT_FOUND_STR)
    logger.debug('\tPyQt ' + PYQT_VERSION_STR if PYQT_VERSION_STR else NOT_FOUND_STR)

    model = ptychodus.model.ModelCore.createInstance(isDeveloperModeEnabled = parsed_args.dev)
    model.start()
    result = 0

    if parsed_args.settings:
        result = model.settingsPresenter.openSettings(parsed_args.settings)

        if result != 0:
            return result

    if parsed_args.batch:
        model.probePresenter.initializeProbe()
        model.objectPresenter.initializeObject()
        result = model.reconstructorPresenter.reconstruct()
    elif PyQt5:
        # QApplication expects the first argument to be the program name
        qt_args = sys.argv[:1] + unparsed_args
        app = QApplication(qt_args)

        view = ptychodus.view.ViewCore.createInstance()
        controller = ptychodus.controller.ControllerCore.createInstance(model, view)

        view.setWindowTitle(version_string())
        view.show()
        result = app.exec_()
    else:
        logger.error('PyQt5 ' + NOT_FOUND_STR)
        result = -1

    model.stop()

    return result


sys.exit(main())

