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


class NSLS2Style1PositionFileReader(ProbePositionsFileReader):
    ONE_MICRON_M: Final[float] = 1.0e-6

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            h5_positions = h5_file['/scan/scan_positions']

            for idx, row in enumerate(h5_positions[()].T):
                point = ProbePosition(
                    idx,
                    row[0] * self.ONE_MICRON_M,
                    row[1] * self.ONE_MICRON_M,
                )
                point_list.append(point)

        return ProbePositionSequence(point_list)


class NSLS2Style2PositionFileReader(ProbePositionsFileReader):
    ONE_MICRON_M: Final[float] = 1.0e-6

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            h5_positions = h5_file['/points']

            for idx, row in enumerate(h5_positions[()].T):
                point = ProbePosition(
                    idx,
                    row[0] * self.ONE_MICRON_M,
                    row[1] * self.ONE_MICRON_M,
                )
                point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_positions_file_readers.register_plugin(
        NSLS2Style1PositionFileReader(),
        simple_name='NSLS_II_1',
        display_name='NSLS-II Style 1 Files (*.h5 *.hdf5)',
    )
    registry.probe_positions_file_readers.register_plugin(
        NSLS2Style2PositionFileReader(),
        simple_name='NSLS_II_2',
        display_name='NSLS-II Style 2 Files (*.h5 *.hdf5)',
    )
