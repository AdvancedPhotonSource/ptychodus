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
    def __init__(self, filePath: Path, index: int) -> None:
        super().__init__()
        self._filePath = filePath
        self._indexes = numpy.array([index])

    def getLabel(self) -> str:
        return self._filePath.stem

    def getIndexes(self) -> PatternIndexesType:
        return self._indexes

    def getData(self) -> PatternDataType:
        with TiffFile(self._filePath) as tiff:
            data = tiff.asarray()

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        return data


class TiffDiffractionFileReader(DiffractionFileReader):
    def _getFileSeries(self, filePath: Path) -> tuple[Mapping[int, Path], str]:
        filePathDict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', filePath.stem)
        longest_digits = max(digits, key=len)
        filePattern = filePath.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in filePath.parent.iterdir():
            z = re.match(filePattern, fp.name)

            if z:
                index = int(z.group(1).lstrip('0'))
                filePathDict[index] = fp

        return filePathDict, filePattern

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        filePathMapping, filePattern = self._getFileSeries(filePath)
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()

        for idx, (_, fp) in enumerate(sorted(filePathMapping.items())):  # TODO use keys
            array = TiffDiffractionPatternArray(fp, idx)
            contentsTree.createChild([array.getLabel(), 'TIFF', str(idx)])
            arrayList.append(array)

        try:
            with TiffFile(filePath) as tiff:
                data = tiff.asarray()
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')
        else:
            if data.ndim == 2:
                data = data[numpy.newaxis, :, :]

            numberOfPatternsPerArray, detectorHeight, detectorWidth = data.shape

            metadata = DiffractionMetadata(
                numberOfPatternsPerArray=numberOfPatternsPerArray,
                numberOfPatternsTotal=numberOfPatternsPerArray * len(arrayList),
                patternDataType=data.dtype,
                detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                filePath=filePath.parent / filePattern,
            )

            dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.register_plugin(
        TiffDiffractionFileReader(),
        simple_name='TIFF',
        display_name='Tagged Image File Format Files (*.tif *.tiff)',
    )


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiffFile = reader.read(filePath)
    print(tiffFile)
