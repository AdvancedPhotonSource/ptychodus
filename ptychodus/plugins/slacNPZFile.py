from pathlib import Path
from typing import Final
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.patterns import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                    DiffractionPatternState, SimpleDiffractionDataset,
                                    SimpleDiffractionPatternArray)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class SLAC_NPZDiffractionFileReader(DiffractionFileReader):

    def read(self, filePath: Path) -> DiffractionDataset:

        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            patterns = npz['diffraction']
            patterns = numpy.transpose(patterns[:, :, :], [2, 0, 1])
        except KeyError:
            logger.warning(f'No diffraction patterns in \"{filePath}\"!')
            return SimpleDiffractionDataset.createNullInstance(filePath)

        numberOfPatterns, detectorHeight, detectorWidth = patterns.shape

        metadata = DiffractionMetadata(
            numberOfPatternsPerArray=numberOfPatterns,
            numberOfPatternsTotal=numberOfPatterns,
            patternDataType=patterns.dtype,
            detectorExtent=ImageExtent(detectorWidth, detectorHeight),
            filePath=filePath,
        )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        contentsTree.createChild(
            [filePath.stem,
             type(patterns).__name__, f'{patterns.dtype}{patterns.shape}'])

        array = SimpleDiffractionPatternArray(
            label=filePath.stem,
            index=0,
            data=patterns,
            state=DiffractionPatternState.FOUND,
        )

        return SimpleDiffractionDataset(metadata, contentsTree, [array])


class SLAC_NPZScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        with numpy.load(filePath) as npzFile:
            scanXInMeters = npzFile['xcoords_start']
            scanYInMeters = npzFile['ycoords_start']

        scanPointList: list[ScanPoint] = list()

        for idx, (x_m, y_m) in enumerate(zip(scanXInMeters, scanYInMeters)):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        return Scan(scanPointList)


class SLAC_NPZProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return Probe()

        try:
            array = npz['probeGuess']
        except KeyError:
            logger.warning(f'No probe guess in \"{filePath}\"!')
            return Probe()

        return Probe(array)


class SLAC_NPZObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> Object:
        try:
            npz = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file \"{filePath}\".')
            return Object()

        try:
            array = npz['objectGuess']
        except KeyError:
            logger.warning(f'No object guess in \"{filePath}\"!')
            return Object()

        return Object(array)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC'
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'

    registry.diffractionFileReaders.registerPlugin(
        SLAC_NPZDiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        SLAC_NPZScanFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        SLAC_NPZProbeFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.objectFileReaders.registerPlugin(
        SLAC_NPZObjectFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
