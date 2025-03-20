from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class NanoMaxPositionFileReader(PositionFileReader):
    MICRONS_TO_METERS: Final[float] = 1.0e-6

    def read(self, file_path: Path) -> PositionSequence:
        pointList: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5File:
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

        return PositionSequence(pointList)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        NanoMaxPositionFileReader(),
        simple_name='NanoMax',
        display_name='NanoMax DiffractionEndStation Files (*.h5 *.hdf5)',
    )
