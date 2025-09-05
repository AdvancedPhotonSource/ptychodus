from pathlib import Path
from typing import Final
import logging
import sys

from scipy.stats import gaussian_kde
import h5py
import matplotlib.pyplot as plt
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

# use full module path to make this file usable as an entry point
from ptychodus.plugins.h5_diffraction_file import H5DiffractionFileReader

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


if __name__ == '__main__':
    file_path = Path(sys.argv[1])

    with h5py.File(file_path, 'r') as h5_file:
        ipm2 = h5_file['/ipm2/sum'][:]

    q0 = numpy.min(ipm2)
    q1, q2, q3 = numpy.quantile(ipm2, [0.25, 0.50, 0.75])
    q4 = numpy.max(ipm2)

    print(f'ipm2: {ipm2.dtype}{ipm2.shape}')
    print(f'min = {q0}')
    print(f'Q1 = {q1}')
    print(f'Q2 = {q2}')
    print(f'Q3 = {q3}')
    print(f'max = {q4}')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ipm2, '.', linewidth=1.5)
    ax.set_title('IPM2')
    ax.set_xlabel('Position')
    ax.set_ylabel('Value')
    ax.grid(True)
    plt.show()

    values = numpy.linspace(q0, q4, 1000)
    kde = gaussian_kde(ipm2)
    density = kde(values)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(values, density, '.-', linewidth=1.5)
    ax.set_title('IPM2')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.grid(True)
    plt.show()
