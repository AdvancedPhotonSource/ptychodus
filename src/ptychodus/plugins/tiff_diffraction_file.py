from collections.abc import Mapping
from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    PatternDataType,
    PatternIndexesType,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionPatternArray(DiffractionPatternArray):
    def __init__(self, file_path: Path, index: int) -> None:
        super().__init__()
        self._file_path = file_path
        self._indexes = numpy.array([index])

    def get_label(self) -> str:
        return self._file_path.stem

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        with TiffFile(self._file_path) as tiff:
            data = tiff.asarray()

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        return data


class TiffDiffractionFileReader(DiffractionFileReader):
    def _getFileSeries(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        file_pathDict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', file_path.stem)
        longest_digits = max(digits, key=len)
        filePattern = file_path.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in file_path.parent.iterdir():
            z = re.match(filePattern, fp.name)

            if z:
                index = int(z.group(1).lstrip('0'))
                file_pathDict[index] = fp

        return file_pathDict, filePattern

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        file_pathMapping, filePattern = self._getFileSeries(file_path)
        contentsTree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, (_, fp) in enumerate(sorted(file_pathMapping.items())):  # TODO use keys
            array = TiffDiffractionPatternArray(fp, idx)
            contentsTree.create_child([array.get_label(), 'TIFF', str(idx)])
            arrayList.append(array)

        try:
            with TiffFile(file_path) as tiff:
                data = tiff.asarray()
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')
        else:
            if data.ndim == 2:
                data = data[numpy.newaxis, :, :]

            numberOfPatternsPerArray, detectorHeight, detectorWidth = data.shape

            metadata = DiffractionMetadata(
                num_patterns_per_array=numberOfPatternsPerArray,
                num_patterns_total=numberOfPatternsPerArray * len(arrayList),
                pattern_dtype=data.dtype,
                detector_extent=ImageExtent(detectorWidth, detectorHeight),
                file_path=file_path.parent / filePattern,
            )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        TiffDiffractionFileReader(),
        simple_name='TIFF',
        display_name='Tagged Image File Format Files (*.tif *.tiff)',
    )


if __name__ == '__main__':
    file_path = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiffFile = reader.read(file_path)
    print(tiffFile)
