from pathlib import Path
import logging

import h5py

from ptychodus.api.patterns import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                    ImageExtent, PixelGeometry, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class LYNXDiffractionFileReader(DiffractionFileReader):

    def __init__(self) -> None:
        self._dataPath = '/entry/data/eiger_4'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                    pixelSize = float(data.attrs['Pixel_size'].item())
                except KeyError:
                    logger.debug('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        detectorPixelGeometry=PixelGeometry(pixelSize, pixelSize),
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
        LYNXDiffractionFileReader(),
        simpleName='LYNX',
        displayName='LYNX Diffraction Files (*.h5 *.hdf5)',
    )
