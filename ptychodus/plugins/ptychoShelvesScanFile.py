from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class PtychoShelvesScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        pointList: list[ScanPoint] = list()

        try:
            with h5py.File(filePath, 'r') as h5File:
                try:
                    ppX = numpy.squeeze(h5File['/ppX'])
                    ppY = numpy.squeeze(h5File['/ppY'])
                except KeyError:
                    logger.debug('Unable to find data.')
                else:
                    if ppX.shape == ppY.shape:
                        logger.debug(f'Coordinate arrays have shape {ppX.shape}.')
                    else:
                        raise ScanPointParseError('Coordinate array shape mismatch!')

                    for idx, (x, y) in enumerate(zip(ppX, ppY)):
                        point = ScanPoint(idx, x, y)
                        pointList.append(point)
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')

        return Scan(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        PtychoShelvesScanFileReader(),
        simpleName='PtychoShelves',
        displayName='PtychoShelves Scan Position Files (*.h5 *.hdf5)')
