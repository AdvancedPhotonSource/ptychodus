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


class NPYDiffractionFileIO(DiffractionFileReader, DiffractionFileWriter):
    SIMPLE_NAME: Final[str] = 'NPY'
    DISPLAY_NAME: Final[str] = 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            data = numpy.load(filePath)
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')
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
                [filePath.stem, type(data).__name__, f'{data.dtype}{data.shape}']
            )

            array = SimpleDiffractionPatternArray(
                label=filePath.stem,
                indexes=numpy.arange(numberOfPatterns),
                data=data,
            )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])

        return dataset

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        patterns = numpy.concatenate([array.getData() for array in dataset])
        numpy.save(filePath, patterns)


def register_plugins(registry: PluginRegistry) -> None:
    npyDiffractionFileIO = NPYDiffractionFileIO()

    registry.diffractionFileReaders.register_plugin(
        npyDiffractionFileIO,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
    registry.diffractionFileWriters.register_plugin(
        npyDiffractionFileIO,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
