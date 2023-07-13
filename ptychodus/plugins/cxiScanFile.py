from pathlib import Path
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan

logger = logging.getLogger(__name__)


class CXIScanFileReader(ScanFileReader):

    @property
    def simpleName(self) -> str:
        return 'CXI'

    @property
    def fileFilter(self) -> str:
        return 'Coherent X-ray Imaging Files (*.cxi)'

    def read(self, filePath: Path) -> Scan:
        pointList = list()

        with h5py.File(filePath, 'r') as h5File:
            try:
                xyzArray = h5File['/entry_1/data_1/translation'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                for xyz in xyzArray:
                    try:
                        x, y, z = xyz
                    except ValueError:
                        logger.exception('Unable to load scan.')
                    else:
                        point = ScanPoint(x, y)
                        pointList.append(point)

        return TabularScan.createFromPointIterable(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CXIScanFileReader())
