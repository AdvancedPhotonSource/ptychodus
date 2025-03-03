from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.product import ELECTRON_VOLT_J
from ptychodus.api.propagator import WavefieldArrayType
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

logger = logging.getLogger(__name__)


class CXIDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._dataPath = '/entry_1/data_1/data'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    detectorExtent = ImageExtent(detectorWidth, detectorHeight)
                    detectorDistanceInMeters = float(
                        h5File['/entry_1/instrument_1/detector_1/distance'][()]
                    )
                    detectorPixelGeometry = PixelGeometry(
                        float(h5File['/entry_1/instrument_1/detector_1/x_pixel_size'][()]),
                        float(h5File['/entry_1/instrument_1/detector_1/y_pixel_size'][()]),
                    )
                    probeEnergyInJoules = float(h5File['/entry_1/instrument_1/source_1/energy'][()])
                    probeEnergyInElectronVolts = probeEnergyInJoules / ELECTRON_VOLT_J

                    # TODO load detector mask; zeros are good pixels
                    # /entry_1/instrument_1/detector_1/mask Dataset {512, 512}

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorDistanceInMeters=detectorDistanceInMeters,
                        detectorExtent=detectorExtent,
                        detectorPixelGeometry=detectorPixelGeometry,
                        probeEnergyInElectronVolts=probeEnergyInElectronVolts,
                        filePath=filePath,
                    )

                    array = H5DiffractionPatternArray(
                        label=filePath.stem,
                        indexes=numpy.arange(numberOfPatterns),
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset


class CXIScanFileReader(ScanFileReader):
    def read(self, filePath: Path) -> Scan:
        scanPointList: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            xyz_m = h5File['/entry_1/data_1/translation'][()]

            for idx, (x, y, z) in enumerate(xyz_m):
                point = ScanPoint(idx, x, y)
                scanPointList.append(point)

        return Scan(scanPointList)


class CXIProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        array: WavefieldArrayType | None = None

        with h5py.File(filePath, 'r') as h5File:
            array = h5File['/entry_1/instrument_1/source_1/illumination'][()]

        return Probe(array=array, pixelGeometry=None)


def registerPlugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'CXI'
    DISPLAY_NAME: Final[str] = 'Coherent X-ray Imaging Files (*.cxi)'

    registry.diffractionFileReaders.registerPlugin(
        CXIDiffractionFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        CXIScanFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        CXIProbeFileReader(),
        simpleName=SIMPLE_NAME,
        displayName=DISPLAY_NAME,
    )
