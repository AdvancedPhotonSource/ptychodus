from pathlib import Path
import logging

import numpy

from ptychodus.api.patterns import (DiffractionDataset, DiffractionFileReader,
                                    DiffractionFileWriter, DiffractionMetadata,
                                    DiffractionPatternState, ImageExtent, SimpleDiffractionDataset,
                                    SimpleDiffractionPatternArray)
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
                detectorExtent=ImageExtent(detectorWidth, detectorHeight),
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


class NPYDiffractionFileWriter(DiffractionFileWriter):

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        data = numpy.concatenate([array.getData() for array in dataset])
        numpy.save(filePath, data)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        NPYDiffractionFileReader(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
    registry.diffractionFileWriters.registerPlugin(
        NPYDiffractionFileWriter(),
        simpleName='NPY',
        displayName='NumPy Binary Files (*.npy)',
    )
