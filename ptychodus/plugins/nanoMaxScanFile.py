from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class NanoMaxScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[float] = 1.0e-6

    def read(self, filePath: Path) -> Scan:
        pointList: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            try:
                positionX = h5File['/entry/measurement/pseudo/x'][()]
                positionY = h5File['/entry/measurement/pseudo/y'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                if positionX.shape == positionY.shape:
                    logger.debug(f'Coordinate arrays have shape {positionX.shape}.')
                else:
                    raise ScanPointParseError('Coordinate array shape mismatch!')

                for idx, (x, y) in enumerate(zip(positionX, positionY)):
                    point = ScanPoint(
                        idx,
                        x * self.MICRONS_TO_METERS,
                        y * self.MICRONS_TO_METERS,
                    )
                    pointList.append(point)

        return Scan(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        NanoMaxScanFileReader(),
        simpleName='NanoMax',
        displayName='NanoMax DiffractionEndStation Scan Files (*.h5 *.hdf5)',
    )
