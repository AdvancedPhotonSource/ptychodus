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
            num_patterns_per_array=numberOfPatterns,
            num_patterns_total=numberOfPatterns,
            pattern_dtype=patterns.dtype,
            detector_extent=ImageExtent(detectorWidth, detectorHeight),
            file_path=filePath,
        )

        contentsTree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        contentsTree.create_child(
            [filePath.stem, type(patterns).__name__, f'{patterns.dtype}{patterns.shape}']
        )

        array = SimpleDiffractionPatternArray(
            label=filePath.stem,
            indexes=numpy.arange(numberOfPatterns),
            data=patterns,
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
            detector_distance_m=0.0,  # not included in file
            probe_energy_eV=0.0,  # not included in file
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
        )

        scanPointList: list[ScanPoint] = list()

        for idx, (x_m, y_m) in enumerate(zip(scanXInMeters, scanYInMeters)):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        costs: Sequence[float] = list()  # not included in file

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=Probe(array=probeArray, pixel_geometry=None),
            object_=Object(array=objectArray, pixel_geometry=None, center=None),
            costs=costs,
        )


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC'
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'

    registry.diffractionFileReaders.register_plugin(
        SLACDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.productFileReaders.register_plugin(
        SLACProductFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
