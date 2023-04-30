from decimal import Decimal
from pathlib import Path
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan

logger = logging.getLogger(__name__)


class PtychoShelvesScanFileReader(ScanFileReader):

    @property
    def simpleName(self) -> str:
        return 'PtychoShelves'

    @property
    def fileFilter(self) -> str:
        return 'PtychoShelves Scan Position Files (*.h5 *.hdf5)'

    def read(self, filePath: Path) -> Scan:
        pointList = list()

        try:
            with h5py.File(filePath, 'r') as h5File:
                try:
                    ppX = h5File['/ppX']
                    ppY = h5File['/ppY']
                except KeyError:
                    logger.debug('Unable to find data.')
                else:
                    if ppX.shape == ppY.shape:
                        logger.debug(f'Coordinate arrays have shape {ppX.shape}.')
                    else:
                        raise ScanPointParseError('Coordinate array shape mismatch!')

                    for xf, yf in zip(ppX[:, 0], ppY[:, 0]):
                        point = ScanPoint(
                            x=Decimal(repr(xf)),
                            y=Decimal(repr(yf)),
                        )
                        pointList.append(point)
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')

        return TabularScan.createFromPointIterable(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(PtychoShelvesScanFileReader())
