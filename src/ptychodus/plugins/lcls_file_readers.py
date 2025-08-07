from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

from .h5_diffraction_file import H5DiffractionFileReader

logger = logging.getLogger(__name__)


class LCLSPositionFileReader(PositionFileReader):
    SIMPLE_NAME: Final[str] = 'LCLS_XPP'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'LCLS X-ray Pump Probe Files (*.h5 *.hdf5)'  # noqa: N806
    ONE_MICRON_M: Final[float] = 1e-6

    def __init__(
        self,
        tomography_angle_deg: float = 180.0,
        ipm2_low_threshold: float = 2000.0,
        ipm2_high_threshold: float = 3000.0,
    ) -> None:
        self._tomography_angle_deg = tomography_angle_deg
        self._ipm2_low_threshold = ipm2_low_threshold
        self._ipm2_high_threshold = ipm2_high_threshold

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5_file:
            # piezo stage positions are in microns
            piezo_stage_position_x_um = h5_file['/lmc/ch03'][:]
            piezo_stage_position_y_um = h5_file['/lmc/ch04'][:]
            piezo_stage_position_z_um = h5_file['/lmc/ch05'][:]

            # ipm2 is used for filtering and normalizing the data
            ipm2 = h5_file['/ipm2/sum'][:]

            # vertical coordinate
            ycoords = -piezo_stage_position_z_um * self.ONE_MICRON_M

            # horizontal coordinate
            tomography_angle_rad = numpy.deg2rad(self._tomography_angle_deg)
            cos_angle = numpy.cos(tomography_angle_rad)
            sin_angle = numpy.sin(tomography_angle_rad)
            xcoords = (
                cos_angle * piezo_stage_position_x_um + sin_angle * piezo_stage_position_y_um
            ) * self.ONE_MICRON_M

            for index, (ipm, x, y) in enumerate(zip(ipm2, xcoords, ycoords)):
                if self._ipm2_low_threshold <= ipm and ipm < self._ipm2_high_threshold:
                    if numpy.isfinite(x) and numpy.isfinite(y):
                        point = ScanPoint(index, x, y)
                        point_list.append(point)
                else:
                    logger.debug(f'Filtered scan point {index=} {ipm=}.')

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        H5DiffractionFileReader(data_path='/jungfrau1M/image_img'),
        simple_name=LCLSPositionFileReader.SIMPLE_NAME,
        display_name=LCLSPositionFileReader.DISPLAY_NAME,
    )
    registry.position_file_readers.register_plugin(
        LCLSPositionFileReader(),
        simple_name=LCLSPositionFileReader.SIMPLE_NAME,
        display_name=LCLSPositionFileReader.DISPLAY_NAME,
    )
