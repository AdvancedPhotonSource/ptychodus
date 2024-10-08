from pathlib import Path
from typing import Final
import logging

import h5py
import numpy
import tables

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionPatternArrayType,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    DiffractionPatternState,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

from .h5DiffractionFile import H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class PyTablesDiffractionPatternArray(DiffractionPatternArray):
    def __init__(self, label: str, index: int, filePath: Path, dataPath: str) -> None:
        super().__init__()
        self._label = label
        self._index = index
        self._state = DiffractionPatternState.UNKNOWN
        self._filePath = filePath
        self._dataPath = dataPath

    def getLabel(self) -> str:
        return self._label

    def getIndex(self) -> int:
        return self._index

    def getState(self) -> DiffractionPatternState:
        return self._state

    def getData(self) -> DiffractionPatternArrayType:
        self._state = DiffractionPatternState.MISSING

        with tables.open_file(self._filePath, mode='r') as h5file:
            try:
                item = h5file.get_node(self._dataPath)
            except tables.NoSuchNodeError:
                raise ValueError(f'Symlink {self._filePath}:{self._dataPath} is broken!')
            else:
                if isinstance(item, tables.EArray):
                    self._state = DiffractionPatternState.FOUND
                else:
                    raise ValueError(
                        f'Symlink {self._filePath}:{self._dataPath} is not a tables File!'
                    )

            data = item[:]

        return data


class LCLSDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._dataPath = '/jungfrau1M/image_img'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)
        metadata = DiffractionMetadata.createNullInstance(filePath)

        try:
            with tables.open_file(filePath, mode='r') as h5File:
                try:
                    data = h5File.get_node(self._dataPath)
                except tables.NoSuchNodeError:
                    logger.debug('Unable to find data.')
                else:
                    data_shape = h5File.root.jungfrau1M.image_img.shape
                    numberOfPatterns, detectorHeight, detectorWidth = data_shape

                    array = PyTablesDiffractionPatternArray(
                        label=filePath.stem,
                        index=0,
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )
                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath,
                    )

            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.debug(f'Unable to read file "{filePath}".')

        return dataset


class LCLSScanFileReader(ScanFileReader):
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

    def read(self, filePath: Path) -> Scan:
        scanPointList: list[ScanPoint] = list()

        with tables.open_file(filePath, mode='r') as h5file:
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

        return Scan(scanPointList)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'LCLS_XPP'

    registry.diffractionFileReaders.registerPlugin(
        LCLSDiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName='LCLS XPP Diffraction Files (*.h5 *.hdf5)',
    )
    registry.scanFileReaders.registerPlugin(
        LCLSScanFileReader(
            tomographyAngleInDegrees=180.0,
            ipm2LowThreshold=2500.0,
            ipm2HighThreshold=6000.0,
        ),
        simpleName=SIMPLE_NAME,
        displayName='LCLS XPP Scan Files (*.h5 *.hdf5)',
    )
