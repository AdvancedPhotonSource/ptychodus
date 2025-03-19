from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class PtychoShelvesPositionFileReader(PositionFileReader):
    def read(self, file_path: Path) -> PositionSequence:
        pointList: list[ScanPoint] = list()

        try:
            with h5py.File(file_path, 'r') as h5File:
                try:
                    ppX = numpy.squeeze(h5File['/ppX'])
                    ppY = numpy.squeeze(h5File['/ppY'])
                except KeyError:
                    logger.warning('Unable to find data.')
                else:
                    if ppX.shape == ppY.shape:
                        logger.debug(f'Coordinate arrays have shape {ppX.shape}.')
                    else:
                        raise ScanPointParseError('Coordinate array shape mismatch!')

                    for idx, (x, y) in enumerate(zip(ppX, ppY)):
                        point = ScanPoint(idx, x, y)
                        pointList.append(point)
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')

        return PositionSequence(pointList)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        PtychoShelvesPositionFileReader(),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Files (*.h5 *.hdf5)',
    )
