from pathlib import Path
import logging

import h5py

from ptychodus.api.constants import LengthUnit
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
)

logger = logging.getLogger(__name__)


class CSSIPositionFileReader(ProbePositionFileReader):
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
                        row[0] * LengthUnit.MILLIMETER.meters_per_unit,
                        row[1] * LengthUnit.MILLIMETER.meters_per_unit,
                    )
                    point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        CSSIPositionFileReader(),
        simple_name='APS_CSSI',
        display_name='APS 9-ID-D CSSI Files (*.h5 *.hdf5)',
    )
