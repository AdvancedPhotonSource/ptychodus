from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan

logger = logging.getLogger(__name__)


class NanoMaxScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[float] = 1.e-6

    def read(self, filePath: Path) -> Scan:
        pointList = list()

        with h5py.File(filePath, 'r') as h5File:
            try:
                xArray = h5File['/entry/measurement/pseudo/x'][()]
                yArray = h5File['/entry/measurement/pseudo/y'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                for x, y in zip(xArray, yArray):
                    point = ScanPoint(
                        x=x * self.MICRONS_TO_METERS,
                        y=y * self.MICRONS_TO_METERS,
                    )
                    pointList.append(point)

        return TabularScan.createFromPointIterable(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        NanoMaxScanFileReader(),
        simpleName='NanoMax',
        displayName='NanoMax DiffractionEndStation Scan Files (*.h5 *.hdf5)',
    )
