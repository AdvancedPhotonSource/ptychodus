from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                    SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class LCLSDiffractionFileReader(DiffractionFileReader):

    def __init__(self, dataPath: str) -> None:
        self._dataPath = dataPath
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.createNullInstance(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.debug('Unable to find data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        filePath=filePath,
                    )

                    array = H5DiffractionPatternArray(
                        label=filePath.stem,
                        index=0,
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
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

        with h5py.File(filePath, 'r') as h5File:
            try:
                # piezo stage positions are in microns
                pi_x = h5File['/lmc/ch03'][()]
                pi_y = h5File['/lmc/ch04'][()]
                pi_z = h5File['/lmc/ch05'][()]

                # ipm2 is used for filtering and normalizing the data
                ipm2 = h5File['/ipm2/sum'][()]
            except KeyError:
                logger.exception('Unable to load scan.')
            else:
                # vertical coordinate is always pi_z
                ycoords = -pi_z * self.MICRONS_TO_METERS

                # horizontal coordinate may be a combination of pi_x and pi_y
                tomographyAngleInRadians = numpy.deg2rad(self._tomographyAngleInDegrees)
                cosAngle = numpy.cos(tomographyAngleInRadians)
                sinAngle = numpy.sin(tomographyAngleInRadians)
                xcoords = (cosAngle * pi_x + sinAngle * pi_y) * self.MICRONS_TO_METERS

                for index, (ipm, x_m, y_m) in enumerate(zip(ipm2, xcoords, ycoords)):
                    if self._ipm2LowThreshold <= ipm and ipm < self._ipm2HighThreshold:
                        if numpy.isfinite(x_m) and numpy.isfinite(y_m):
                            point = ScanPoint(index, x_m, y_m)
                            scanPointList.append(point)
                    else:
                        logger.debug(f'Filtered scan point {index=} {ipm=}.')

        return Scan(scanPointList)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'LCLS'

    registry.diffractionFileReaders.registerPlugin(
        LCLSDiffractionFileReader('/jungfrau1M/ROI_0_area'),
        simpleName=SIMPLE_NAME,
        displayName='LCLS Diffraction Files (*.h5 *.hdf5)',
    )
    registry.scanFileReaders.registerPlugin(
        LCLSScanFileReader(
            tomographyAngleInDegrees=180.,
            ipm2LowThreshold=40000.,
            ipm2HighThreshold=50000.,
        ),
        simpleName=SIMPLE_NAME,
        displayName='LCLS Scan Files (*.h5 *.hdf5)',
    )
