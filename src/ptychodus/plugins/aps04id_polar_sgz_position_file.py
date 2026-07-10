from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class PolarSoftGlueZynqPositionFileReader(ProbePositionFileReader):
    ONE_NANOMETER_M: Final[float] = 1.0e-9
    DATA_PATH: Final[str] = '/entry/data/data'
    COLUMN_X: Final[int] = 3
    COLUMN_Y: Final[int] = 4

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                data = h5_file[self.DATA_PATH][()]
            except KeyError as ex:
                raise ProbePositionParseError(f'Missing dataset {self.DATA_PATH!r}.') from ex

            if data.ndim != 2 or data.shape[1] <= max(self.COLUMN_X, self.COLUMN_Y):
                raise ProbePositionParseError(
                    f'Unexpected dataset shape {data.shape} at {self.DATA_PATH}.'
                )

            position_x = data[:, self.COLUMN_X]
            position_y = data[:, self.COLUMN_Y]
            logger.debug(f'Coordinate arrays have shape {position_x.shape}.')

            for idx, (x, y) in enumerate(zip(position_x, position_y)):
                point_list.append(
                    ProbePosition(
                        idx,
                        float(x) * self.ONE_NANOMETER_M,
                        float(y) * self.ONE_NANOMETER_M,
                    )
                )

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        PolarSoftGlueZynqPositionFileReader(),
        simple_name='APS_Polar_SGZ',
        display_name='APS 4-ID Polar softGlueZynq Files (*.h5)',
    )
