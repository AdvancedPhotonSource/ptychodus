from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class NanoMaxPositionFileReader(PositionFileReader):
    ONE_MICRON_M: Final[float] = 1.0e-6

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                position_x = h5_file['/entry/measurement/pseudo/x'][()]
                position_y = h5_file['/entry/measurement/pseudo/y'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                if position_x.shape == position_y.shape:
                    logger.debug(f'Coordinate arrays have shape {position_x.shape}.')
                else:
                    raise ScanPointParseError('Coordinate array shape mismatch!')

                for idx, (x, y) in enumerate(zip(position_x, position_y)):
                    point = ScanPoint(
                        idx,
                        x * self.ONE_MICRON_M,
                        y * self.ONE_MICRON_M,
                    )
                    point_list.append(point)

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        NanoMaxPositionFileReader(),
        simple_name='MAX_IV_NanoMAX',
        display_name='MAX IV NanoMAX Diffraction Endstation Files (*.h5 *.hdf5)',
    )
