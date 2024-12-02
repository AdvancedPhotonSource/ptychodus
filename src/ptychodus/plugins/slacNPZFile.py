from pathlib import Path
from typing import Final, Sequence
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternState,
    SimpleDiffractionDataset,
    SimpleDiffractionPatternArray,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.scan import Scan, ScanPoint
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class SLACDiffractionFileReader(DiffractionFileReader):
    def read(self, filePath: Path) -> DiffractionDataset:
        with numpy.load(filePath) as npzFile:
            patterns = numpy.transpose(npzFile['diffraction'], [2, 0, 1])

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
            [filePath.stem, type(patterns).__name__, f'{patterns.dtype}{patterns.shape}']
        )

        array = SimpleDiffractionPatternArray(
            label=filePath.stem,
            index=0,
            data=patterns,
            state=DiffractionPatternState.FOUND,
        )

        return SimpleDiffractionDataset(metadata, contentsTree, [array])


class SLACProductFileReader(ProductFileReader):
    def read(self, filePath: Path) -> Product:
        with numpy.load(filePath) as npzFile:
            scanXInMeters = npzFile['xcoords_start']
            scanYInMeters = npzFile['ycoords_start']
            probeArray = npzFile['probeGuess']
            objectArray = npzFile['objectGuess']

        metadata = ProductMetadata(
            name=filePath.stem,
            comments='',
            detectorDistanceInMeters=0.0,  # not included in file
            probeEnergyInElectronVolts=0.0,  # not included in file
            probePhotonCount=0.0,  # not included in file
            exposureTimeInSeconds=0.0,  # not included in file
        )

        scanPointList: list[ScanPoint] = list()

        for idx, (x_m, y_m) in enumerate(zip(scanXInMeters, scanYInMeters)):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        costs: Sequence[float] = list()  # not included in file

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=Probe(array=probeArray, pixelGeometry=None),
            object_=Object(array=objectArray, pixelGeometry=None, center=None),
            costs=costs,
        )


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC'
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'

    registry.diffractionFileReaders.registerPlugin(
        SLACDiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.productFileReaders.registerPlugin(
        SLACProductFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
