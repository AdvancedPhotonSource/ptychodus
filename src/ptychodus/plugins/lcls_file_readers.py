from pathlib import Path
from typing import Final
import logging

import h5py
import numpy
import tables

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

from .h5_diffraction_file import H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class PyTablesDiffractionPatternArray(DiffractionPatternArray):
    def __init__(self, label: str, numberOfPatterns: int, file_path: Path, data_path: str) -> None:
        super().__init__()
        self._label = label
        self._indexes = numpy.arange(numberOfPatterns)
        self._file_path = file_path
        self._data_path = data_path

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        with tables.open_file(self._file_path, mode='r') as h5file:
            try:
                item = h5file.get_node(self._data_path)
            except tables.NoSuchNodeError:
                raise ValueError(f'Symlink {self._file_path}:{self._data_path} is broken!')
            else:
                if not isinstance(item, tables.EArray):
                    raise ValueError(
                        f'Symlink {self._file_path}:{self._data_path} is not a tables File!'
                    )

            data = item[:]

        return data


class LCLSDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/jungfrau1M/image_img'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)
        metadata = DiffractionMetadata.create_null(file_path)

        try:
            with tables.open_file(file_path, mode='r') as h5File:
                try:
                    data = h5File.get_node(self._data_path)
                except tables.NoSuchNodeError:
                    logger.debug('Unable to find data.')
                else:
                    data_shape = h5File.root.jungfrau1M.image_img.shape
                    numberOfPatterns, detectorHeight, detectorWidth = data_shape

                    array = PyTablesDiffractionPatternArray(
                        label=file_path.stem,
                        numberOfPatterns=numberOfPatterns,
                        file_path=file_path,
                        data_path=self._data_path,
                    )
                    metadata = DiffractionMetadata(
                        num_patterns_per_array=numberOfPatterns,
                        num_patterns_total=numberOfPatterns,
                        pattern_dtype=data.dtype,
                        detector_extent=ImageExtent(detectorWidth, detectorHeight),
                        file_path=file_path,
                    )

            with h5py.File(file_path, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.debug(f'Unable to read file "{file_path}".')

        return dataset


class LCLSPositionFileReader(PositionFileReader):
    MICRONS_TO_METERS: Final[float] = 1e-6

    def __init__(
        self,
        tomographyAngleInDegrees: float,
        ipm2LowThreshold: float,
        ipm2HighThreshold: float,
    ) -> None:
        self._tomographyAngleInDegrees = tomographyAngleInDegrees
        self._ipm2LowThreshold = ipm2LowThreshold
        self._ipm2HighThreshold = ipm2HighThreshold

    def read(self, file_path: Path) -> PositionSequence:
        scanPointList: list[ScanPoint] = list()

        with tables.open_file(file_path, mode='r') as h5file:
            try:
                # piezo stage positions are in microns
                pi_x = h5file.get_node('/lmc/ch03')[:]
                pi_y = h5file.get_node('/lmc/ch04')[:]
                pi_z = h5file.get_node('/lmc/ch05')[:]

                # ipm2 is used for filtering and normalizing the data
                ipm2 = h5file.get_node('/ipm2/sum')[:]
            except tables.NoSuchNodeError:
                logger.exception('Unable to load scan.')
            else:
                # vertical coordinate is always pi_z
                ycoords = -pi_z * self.MICRONS_TO_METERS

                # horizontal coordinate may be a combination of pi_x and pi_y
                tomographyAngleInRadians = numpy.deg2rad(self._tomographyAngleInDegrees)
                cosAngle = numpy.cos(tomographyAngleInRadians)
                sinAngle = numpy.sin(tomographyAngleInRadians)
                xcoords = (cosAngle * pi_x + sinAngle * pi_y) * self.MICRONS_TO_METERS

                for index, (ipm, x, y) in enumerate(zip(ipm2, xcoords, ycoords)):
                    if self._ipm2LowThreshold <= ipm and ipm < self._ipm2HighThreshold:
                        if numpy.isfinite(x) and numpy.isfinite(y):
                            point = ScanPoint(index, x, y)
                            scanPointList.append(point)
                    else:
                        logger.debug(f'Filtered scan point {index=} {ipm=}.')

        return PositionSequence(scanPointList)


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'LCLS_XPP'

    registry.diffraction_file_readers.register_plugin(
        LCLSDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name='LCLS XPP Files (*.h5 *.hdf5)',
    )
    registry.position_file_readers.register_plugin(
        LCLSPositionFileReader(
            tomographyAngleInDegrees=180.0,
            ipm2LowThreshold=2500.0,
            ipm2HighThreshold=6000.0,
        ),
        simple_name=SIMPLE_NAME,
        display_name='LCLS XPP Files (*.h5 *.hdf5)',
    )
