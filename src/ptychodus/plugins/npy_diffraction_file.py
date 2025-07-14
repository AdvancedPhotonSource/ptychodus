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


class NPYDiffractionFileIO(DiffractionFileReader, DiffractionFileWriter):
    SIMPLE_NAME: Final[str] = 'NPY'
    DISPLAY_NAME: Final[str] = 'NumPy Binary Files (*.npy)'

    def read(self, file_path: Path) -> DiffractionDataset:
        data = numpy.load(file_path)

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        num_patterns, detector_height, detector_width = data.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=data.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path,
        )

        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        contents_tree.create_child(
            [file_path.stem, type(data).__name__, f'{data.dtype}{data.shape}']
        )

        array = SimpleDiffractionArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            patterns=data,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array])

    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        patterns = numpy.concatenate([array.get_patterns() for array in dataset])
        numpy.save(file_path, patterns)


def register_plugins(registry: PluginRegistry) -> None:
    npy_diffraction_file_io = NPYDiffractionFileIO()

    registry.diffraction_file_readers.register_plugin(
        npy_diffraction_file_io,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
    registry.diffraction_file_writers.register_plugin(
        npy_diffraction_file_io,
        simple_name=NPYDiffractionFileIO.SIMPLE_NAME,
        display_name=NPYDiffractionFileIO.DISPLAY_NAME,
    )
