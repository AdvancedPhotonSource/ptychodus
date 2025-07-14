from pathlib import Path
from typing import Final
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionFileWriter,
    DiffractionMetadata,
    SimpleDiffractionDataset,
    SimpleDiffractionArray,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPZDiffractionFileIO(DiffractionFileReader, DiffractionFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'Ptychodus NumPy Zipped Archive (*.npz)'

    INDEXES: Final[str] = 'indexes'
    PATTERNS: Final[str] = 'patterns'
    BAD_PIXELS: Final[str] = 'bad_pixels'

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)
        contents = numpy.load(file_path)

        try:
            patterns = contents[self.PATTERNS]
        except KeyError:
            logger.warning(f'Failed to read patterns in "{file_path}".')
            return dataset

        num_patterns, detector_height, detector_width = patterns.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
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

        array = SimpleDiffractionArray(
            label=file_path.stem,
            indexes=indexes,
            patterns=patterns,
        )

        return SimpleDiffractionDataset(
            metadata, contents_tree, [array], bad_pixels=contents.get(self.BAD_PIXELS, None)
        )

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        contents = {
            self.INDEXES: numpy.concatenate([array.get_indexes() for array in dataset]),
            self.PATTERNS: numpy.concatenate([array.get_patterns() for array in dataset]),
        }

        bad_pixels = dataset.get_bad_pixels()

        if bad_pixels is not None:
            contents[self.BAD_PIXELS] = bad_pixels

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)


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
