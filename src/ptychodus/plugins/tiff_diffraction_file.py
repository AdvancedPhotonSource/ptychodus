from collections.abc import Mapping
from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    DiffractionPatterns,
    DiffractionIndexes,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionPatternArray(DiffractionArray):
    def __init__(self, file_path: Path, index: int) -> None:
        super().__init__()
        self._file_path = file_path
        self._indexes = numpy.array([index])

    def get_label(self) -> str:
        return self._file_path.stem

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        with TiffFile(self._file_path) as tiff:
            data = tiff.asarray()

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        return data


class TiffDiffractionFileReader(DiffractionFileReader):
    def _get_file_series(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        file_path_dict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', file_path.stem)
        longest_digits = max(digits, key=len)
        file_pattern = file_path.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in file_path.parent.iterdir():
            z = re.match(file_pattern, fp.name)

            if z:
                index = int(z.group(1).lstrip('0'))
                file_path_dict[index] = fp

        return file_path_dict, file_pattern

    def read(self, file_path: Path) -> DiffractionDataset:
        file_path_mapping, file_pattern = self._get_file_series(file_path)
        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        array_list: list[DiffractionArray] = list()

        for idx, (_, fp) in enumerate(sorted(file_path_mapping.items())):  # TODO use keys
            array = TiffDiffractionPatternArray(fp, idx)
            contents_tree.create_child([array.get_label(), 'TIFF', str(idx)])
            array_list.append(array)

        with TiffFile(file_path) as tiff:
            data = tiff.asarray()

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        num_patterns_per_array, detector_height, detector_width = data.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns_per_array] * len(array_list),
            pattern_dtype=data.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path.parent / file_pattern,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        TiffDiffractionFileReader(),
        simple_name='TIFF',
        display_name='Tagged Image File Format Files (*.tif *.tiff)',
    )


if __name__ == '__main__':
    file_path = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiff_file = reader.read(file_path)
    print(tiff_file)
