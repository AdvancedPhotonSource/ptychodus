from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionsFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class PtychoShelvesPositionFileReader(ProbePositionsFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                pp_x = numpy.squeeze(h5_file['/ppX'])
                pp_y = numpy.squeeze(h5_file['/ppY'])
            except KeyError:
                logger.warning('Unable to find data.')
            else:
                if pp_x.shape == pp_y.shape:
                    logger.debug(f'Coordinate arrays have shape {pp_x.shape}.')
                else:
                    raise ProbePositionParseError('Coordinate array shape mismatch!')

                for idx, (x, y) in enumerate(zip(pp_x, pp_y)):
                    point = ProbePosition(idx, x, y)
                    point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_positions_file_readers.register_plugin(
        PtychoShelvesPositionFileReader(),
        simple_name='PtychoShelves',
        display_name='PtychoShelves Files (*.h5 *.hdf5)',
    )
