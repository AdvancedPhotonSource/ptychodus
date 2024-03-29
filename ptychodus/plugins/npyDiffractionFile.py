from pathlib import Path
import logging

import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternState, SimpleDiffractionDataset,
                                SimpleDiffractionPatternArray)
from ptychodus.api.image import ImageExtent
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPYDiffractionFileReader(DiffractionFileReader):

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            data = numpy.load(filePath)
        except OSError:
            logger.debug(f'Unable to read file \"{filePath}\".')
        else:
            if data.ndim == 2:
                data = data[numpy.newaxis, :, :]

            numberOfPatterns, detectorHeight, detectorWidth = data.shape

            metadata = DiffractionMetadata(
                numberOfPatternsPerArray=numberOfPatterns,
                numberOfPatternsTotal=numberOfPatterns,
                patternDataType=data.dtype,
                detectorExtentInPixels=ImageExtent(detectorWidth, detectorHeight),
                filePath=filePath,
            )

            contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
            contentsTree.createChild(
                [filePath.stem, type(data).__name__, f'{data.dtype}{data.shape}'])

            array = SimpleDiffractionPatternArray(
                label=filePath.stem,
                index=0,
                data=data,
                state=DiffractionPatternState.FOUND,
            )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])

        return dataset


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        NPYDiffractionFileReader(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
