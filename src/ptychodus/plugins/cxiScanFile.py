from pathlib import Path
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

logger = logging.getLogger(__name__)


class CXIScanFileReader(ScanFileReader):
    def read(self, filePath: Path) -> Scan:
        pointList: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            try:
                xyzArray = h5File['/entry_1/data_1/translation'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                for idx, xyz in enumerate(xyzArray):
                    try:
                        x, y, z = xyz
                    except ValueError:
                        logger.exception(f'Unable to load scan point {xyz=}.')
                    else:
                        point = ScanPoint(idx, x, y)
                        pointList.append(point)

        return Scan(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        CXIScanFileReader(),
        simpleName='CXI',
        displayName='Coherent X-ray Imaging Files (*.cxi)',
    )
