from pathlib import Path
import logging

import h5py

from ptychodus.api.apparatus import ImageExtent, PixelGeometry
from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

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
                    logger.debug('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    detectorExtentInPixels = ImageExtent(detectorWidth, detectorHeight)
                    detectorDistanceInMeters = float(
                        h5File['/entry_1/instrument_1/detector_1/distance'][()])
                    detectorPixelGeometry = PixelGeometry(
                        float(h5File['/entry_1/instrument_1/detector_1/x_pixel_size'][()]),
                        float(h5File['/entry_1/instrument_1/detector_1/y_pixel_size'][()]),
                    )
                    probeEnergyInJoules = float(
                        h5File['/entry_1/instrument_1/source_1/energy'][()])
                    oneJouleInElectronVolts = 6.241509074e18
                    probeEnergyInElectronVolts = probeEnergyInJoules * oneJouleInElectronVolts

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorDistanceInMeters=detectorDistanceInMeters,
                        detectorExtentInPixels=detectorExtentInPixels,
                        detectorPixelGeometry=detectorPixelGeometry,
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
    registry.diffractionFileReaders.registerPlugin(
        CXIDiffractionFileReader(),
        simpleName='CXI',
        displayName='Coherent X-ray Imaging Files (*.cxi)',
    )
