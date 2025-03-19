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

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            data = numpy.load(file_path)
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')
        else:
            if data.ndim == 2:
                data = data[numpy.newaxis, :, :]

            numberOfPatterns, detectorHeight, detectorWidth = data.shape

            metadata = DiffractionMetadata(
                num_patterns_per_array=numberOfPatterns,
                num_patterns_total=numberOfPatterns,
                pattern_dtype=data.dtype,
                detector_extent=ImageExtent(detectorWidth, detectorHeight),
                file_path=file_path,
            )

            contentsTree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
            contentsTree.create_child(
                [file_path.stem, type(data).__name__, f'{data.dtype}{data.shape}']
            )

            array = SimpleDiffractionPatternArray(
                label=file_path.stem,
                indexes=numpy.arange(numberOfPatterns),
                data=data,
            )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])

        return dataset

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        patterns = numpy.concatenate([array.get_data() for array in dataset])
        numpy.save(file_path, patterns)


def register_plugins(registry: PluginRegistry) -> None:
    npyDiffractionFileIO = NPYDiffractionFileIO()

    registry.diffraction_file_readers.register_plugin(
        npyDiffractionFileIO,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
    registry.diffraction_file_writers.register_plugin(
        npyDiffractionFileIO,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
