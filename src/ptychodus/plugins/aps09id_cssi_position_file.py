from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionsFileReader,
    ProbePosition,
)

logger = logging.getLogger(__name__)


class CSSIPositionFileReader(ProbePositionsFileReader):
    ONE_MILLIMETER_M: Final[float] = 1e-3

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                h5_positions = h5_file['/exchange/motor_pos']
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                for idx, row in enumerate(h5_positions):
                    point = ProbePosition(
                        idx,
                        row[0] * self.ONE_MILLIMETER_M,
                        row[1] * self.ONE_MILLIMETER_M,
                    )
                    point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_positions_file_readers.register_plugin(
        CSSIPositionFileReader(),
        simple_name='APS_CSSI',
        display_name='APS 9-ID CSSI Files (*.h5 *.hdf5)',
    )
