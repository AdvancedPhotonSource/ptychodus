#LCLSh5filev2
from pathlib import Path
from typing import Final
import logging

import h5py
import numpy
import tables

from ptychodus.api.patterns import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                    SimpleDiffractionDataset)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

from .h5DiffractionFile import H5DiffractionFileTreeBuilder
from .h5tablesDiffractionFile import H5DiffractionPatternArray

logger = logging.getLogger(__name__)


class LCLSDiffractionFileReader(DiffractionFileReader):

    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with tables.open_file(filePath, mode='r') as h5File:

                try:
                    data = h5File.get_node(self._dataPath)
                except tables.NoSuchNodeError:
                    logger.debug('Unable to find data.')
                else:
                    #Getting shape tuple using tables data reading method
                    data_shape = h5File.root.jungfrau1M.image_img.shape

                    numberOfPatterns, detectorHeight, detectorWidth = data_shape

                    array = H5DiffractionPatternArray(
                        label=filePath.stem,
                        index=0,
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )

                    logger.debug(f'Read diffraction data at {self._dataPath}')

            #Metadata stuff preserved from lclsh5file, just keep this
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.createNullInstance(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.debug('Unable to find data.')
                else:
                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath,
                    )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
            logger.debug('loaded dataset')
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')

        return dataset


class LCLSScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[float] = 1e-6

    def __init__(self, tomographyAngleInDegrees: float, ipm2LowThreshold: float,
                 ipm2HighThreshold: float) -> None:
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
    SIMPLE_NAME: Final[str] = 'LCLSv2'

    registry.diffractionFileReaders.registerPlugin(
        LCLSDiffractionFileReader('/jungfrau1M/image_img'),
        simpleName=SIMPLE_NAME,
        displayName='LCLS h5 Tables Diffraction Files (*.h5 *.hdf5)',
    )
    registry.scanFileReaders.registerPlugin(
        LCLSScanFileReader(
            tomographyAngleInDegrees=180.,
            ipm2LowThreshold=2500.,
            ipm2HighThreshold=6000.,
        ),
        simpleName=SIMPLE_NAME,
        displayName='LCLS h5 Tables Scan Files (*.h5 *.hdf5)',
    )
