from pathlib import Path
import logging

import h5py

from ptychodus.api.constants import ONE_MICRON_M
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class NanoMAXPositionFileReader(ProbePositionFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

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
                    raise ProbePositionParseError('Coordinate array shape mismatch!')

                for idx, (x, y) in enumerate(zip(position_x, position_y)):
                    point = ProbePosition(
                        idx,
                        x * ONE_MICRON_M,
                        y * ONE_MICRON_M,
                    )
                    point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        NanoMAXPositionFileReader(),
        simple_name='MAX_IV_NanoMAX',
        display_name='MAX IV NanoMAX Files (*.h5 *.hdf5)',
    )
