from pathlib import Path
from typing import Final
import logging

import h5py
import numpy
import tables

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    DiffractionPatterns,
    DiffractionIndexes,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

from .h5_diffraction_file import H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class PyTablesDiffractionPatternArray(DiffractionArray):
    def __init__(self, label: str, num_patterns: int, file_path: Path, data_path: str) -> None:
        super().__init__()
        self._label = label
        self._indexes = numpy.arange(num_patterns)
        self._file_path = file_path
        self._data_path = data_path

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        with tables.open_file(str(self._file_path), mode='r') as h5_file:
            item = h5_file.get_node(self._data_path)

            if isinstance(item, tables.EArray):
                return item[:]

        raise ValueError(f'Symlink {self._file_path}:{self._data_path} is not a tables file!')


class LCLSDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/jungfrau1M/image_img'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata.create_null(file_path)
        array_list: list[DiffractionArray] = list()

        with tables.open_file(str(file_path), mode='r') as h5_file:
            data = h5_file.get_node(self._data_path)
            data_shape = h5_file.root.jungfrau1M.image_img.shape
            num_patterns, detector_height, detector_width = data_shape

            array = PyTablesDiffractionPatternArray(
                label=file_path.stem,
                num_patterns=num_patterns,
                file_path=file_path,
                data_path=self._data_path,
            )
            array_list.append(array)
            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data.dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                file_path=file_path,
            )

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


class LCLSPositionFileReader(PositionFileReader):
    ONE_MICRON_M: Final[float] = 1e-6

    def __init__(
        self,
        tomography_angle_deg: float,
        ipm2_low_threshold: float,
        ipm2_high_threshold: float,
    ) -> None:
        self._tomography_angle_deg = tomography_angle_deg
        self._ipm2_low_threshold = ipm2_low_threshold
        self._ipm2_high_threshold = ipm2_high_threshold

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        with tables.open_file(str(file_path), mode='r') as h5_file:
            try:
                # piezo stage positions are in microns
                pi_x = h5_file.get_node('/lmc/ch03')[:]
                pi_y = h5_file.get_node('/lmc/ch04')[:]
                pi_z = h5_file.get_node('/lmc/ch05')[:]

                # ipm2 is used for filtering and normalizing the data
                ipm2 = h5_file.get_node('/ipm2/sum')[:]
            except tables.NoSuchNodeError:
                logger.exception('Unable to load scan.')
            else:
                # vertical coordinate is always pi_z
                ycoords = -pi_z * self.ONE_MICRON_M

                # horizontal coordinate may be a combination of pi_x and pi_y
                tomography_angle_rad = numpy.deg2rad(self._tomography_angle_deg)
                cos_angle = numpy.cos(tomography_angle_rad)
                sin_angle = numpy.sin(tomography_angle_rad)
                xcoords = (cos_angle * pi_x + sin_angle * pi_y) * self.ONE_MICRON_M

                for index, (ipm, x, y) in enumerate(zip(ipm2, xcoords, ycoords)):
                    if self._ipm2_low_threshold <= ipm and ipm < self._ipm2_high_threshold:
                        if numpy.isfinite(x) and numpy.isfinite(y):
                            point = ScanPoint(index, x, y)
                            point_list.append(point)
                    else:
                        logger.debug(f'Filtered scan point {index=} {ipm=}.')

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'LCLS_XPP'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'LCLS X-ray Pump Probe Files (*.h5 *.hdf5)'  # noqa: N806

    registry.diffraction_file_readers.register_plugin(
        LCLSDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.position_file_readers.register_plugin(
        LCLSPositionFileReader(
            tomography_angle_deg=180.0,
            ipm2_low_threshold=2000.0,
            ipm2_high_threshold=3000.0,
        ),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
