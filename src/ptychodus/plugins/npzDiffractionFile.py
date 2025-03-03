from pathlib import Path
from typing import Final
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    SimpleDiffractionDataset,
    SimpleDiffractionPatternArray,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPZDiffractionFileIO(DiffractionFileReader, DiffractionFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'NumPy Zipped Archive (*.npz)'

    INDEXES: Final[str] = 'indexes'
    PATTERNS: Final[str] = 'patterns'

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            contents = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')
            return dataset

        try:
            patterns = contents[self.PATTERNS]
        except KeyError:
            logger.warning(f'Failed to read patterns in "{filePath}".')
            return dataset

        numberOfPatterns, detectorHeight, detectorWidth = patterns.shape

        metadata = DiffractionMetadata(
            numberOfPatternsPerArray=numberOfPatterns,
            numberOfPatternsTotal=numberOfPatterns,
            patternDataType=patterns.dtype,
            detectorExtent=ImageExtent(detectorWidth, detectorHeight),
            filePath=filePath,
        )

        try:
            indexes = contents[self.INDEXES]
        except KeyError:
            logger.warning(f'Failed to read indexes in "{filePath}".')
            indexes = numpy.arange(numberOfPatterns)

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        contentsTree.createChild(
            [
                filePath.stem,
                type(patterns).__name__,
                f'{patterns.dtype}{patterns.shape}',
            ]
        )

        array = SimpleDiffractionPatternArray(
            label=filePath.stem,
            indexes=indexes,
            data=patterns,
        )

        return SimpleDiffractionDataset(metadata, contentsTree, [array])

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        contents = {
            self.INDEXES: numpy.concatenate([array.getIndexes() for array in dataset]),
            self.PATTERNS: numpy.concatenate([array.getData() for array in dataset]),
        }
        numpy.savez(filePath, **contents)


def registerPlugins(registry: PluginRegistry) -> None:
    npzDiffractionFileIO = NPZDiffractionFileIO()

    registry.diffractionFileReaders.registerPlugin(
        npzDiffractionFileIO,
        simpleName=NPZDiffractionFileIO.SIMPLE_NAME,
        displayName=NPZDiffractionFileIO.DISPLAY_NAME,
    )
    registry.diffractionFileWriters.registerPlugin(
        npzDiffractionFileIO,
        simpleName=NPZDiffractionFileIO.SIMPLE_NAME,
        displayName=NPZDiffractionFileIO.DISPLAY_NAME,
    )
