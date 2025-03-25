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

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            contents = numpy.load(file_path)
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')
            return dataset

        try:
            patterns = contents[self.PATTERNS]
        except KeyError:
            logger.warning(f'Failed to read patterns in "{file_path}".')
            return dataset

        num_patterns, detector_height, detector_width = patterns.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns,
            num_patterns_total=num_patterns,
            pattern_dtype=patterns.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path,
        )

        try:
            indexes = contents[self.INDEXES]
        except KeyError:
            logger.warning(f'Failed to read indexes in "{file_path}".')
            indexes = numpy.arange(num_patterns)

        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        contents_tree.create_child(
            [
                file_path.stem,
                type(patterns).__name__,
                f'{patterns.dtype}{patterns.shape}',
            ]
        )

        array = SimpleDiffractionPatternArray(
            label=file_path.stem,
            indexes=indexes,
            data=patterns,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array])

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        contents = {
            self.INDEXES: numpy.concatenate([array.get_indexes() for array in dataset]),
            self.PATTERNS: numpy.concatenate([array.get_data() for array in dataset]),
        }
        numpy.savez(file_path, **contents)


def register_plugins(registry: PluginRegistry) -> None:
    npz_diffraction_file_io = NPZDiffractionFileIO()

    registry.diffraction_file_readers.register_plugin(
        npz_diffraction_file_io,
        simple_name=NPZDiffractionFileIO.SIMPLE_NAME,
        display_name=NPZDiffractionFileIO.DISPLAY_NAME,
    )
    registry.diffraction_file_writers.register_plugin(
        npz_diffraction_file_io,
        simple_name=NPZDiffractionFileIO.SIMPLE_NAME,
        display_name=NPZDiffractionFileIO.DISPLAY_NAME,
    )
