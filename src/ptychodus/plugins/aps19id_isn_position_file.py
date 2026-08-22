from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.constants import LengthUnit
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class ISNPositionFileReader(ProbePositionFileReader):
    """Reader for APS 19-ID-E In-situ Nanoprobe processed position files.

    The position file stores one sample position per detector trigger in
    ``/entry/data/{X_Position,Y_Position}`` (micrometers). Positions are
    paired with diffraction frames by array order, so this reader indexes
    them sequentially rather than by the ``Trigger`` dataset.
    """

    X_POSITION_PATH: Final[str] = '/entry/data/X_Position'
    Y_POSITION_PATH: Final[str] = '/entry/data/Y_Position'

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                position_x = h5_file[self.X_POSITION_PATH][()]
                position_y = h5_file[self.Y_POSITION_PATH][()]
            except KeyError as ex:
                raise ProbePositionParseError(f'Missing position dataset in "{file_path}".') from ex

            if position_x.shape != position_y.shape:
                raise ProbePositionParseError('Coordinate array shape mismatch!')

        for idx, (x, y) in enumerate(zip(position_x, position_y)):
            point_list.append(
                ProbePosition(
                    idx,
                    LengthUnit.MICROMETER.to_meters(float(x)),
                    LengthUnit.MICROMETER.to_meters(float(y)),
                )
            )

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        ISNPositionFileReader(),
        simple_name='APS_ISN',
        display_name='APS 19-ID-E In-situ Nanoprobe Files (*.h5 *.hdf5)',
    )
