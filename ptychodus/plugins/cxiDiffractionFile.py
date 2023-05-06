from decimal import Decimal
from pathlib import Path
import logging

import h5py

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                SimpleDiffractionDataset)
from ptychodus.api.geometry import Array2D
from ptychodus.api.plugins import PluginRegistry
from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class CXIDiffractionFileReader(DiffractionFileReader):

    def __init__(self) -> None:
        self._dataPath = '/entry_1/data_1/data'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    @property
    def simpleName(self) -> str:
        return 'CXI'

    @property
    def fileFilter(self) -> str:
        return 'Coherent X-ray Imaging Files (*.cxi)'

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                except KeyError:
                    logger.debug('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    detectorNumberOfPixels = Array2D[int](detectorWidth, detectorHeight)
                    detectorDistanceInMeters = Decimal(
                        repr(h5File['/entry_1/instrument_1/detector_1/distance'][()]))
                    detectorPixelSizeInMeters = Array2D[Decimal](
                        Decimal(repr(h5File['/entry_1/instrument_1/detector_1/x_pixel_size'][()])),
                        Decimal(repr(h5File['/entry_1/instrument_1/detector_1/y_pixel_size'][()])),
                    )
                    probeEnergyInJoules = Decimal(
                        repr(h5File['/entry_1/instrument_1/source_1/energy'][()]))
                    oneJouleInElectronVolts = Decimal('6.241509074e18')
                    probeEnergyInElectronVolts = probeEnergyInJoules * oneJouleInElectronVolts

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorDistanceInMeters=detectorDistanceInMeters,
                        detectorNumberOfPixels=detectorNumberOfPixels,
                        detectorPixelSizeInMeters=detectorPixelSizeInMeters,
                        probeEnergyInElectronVolts=probeEnergyInElectronVolts,
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


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CXIDiffractionFileReader())
