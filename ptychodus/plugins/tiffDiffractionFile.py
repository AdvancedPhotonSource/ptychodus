from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternArray, DiffractionPatternData,
                                DiffractionPatternState, SimpleDiffractionDataset,
                                SimpleDiffractionPatternArray)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionPatternArray(DiffractionPatternArray):

    def __init__(self, filePath: Path, index: int) -> None:
        super().__init__()
        self._filePath = filePath
        self._index = index
        self._state = DiffractionPatternState.UNKNOWN

    def getLabel(self) -> str:
        return self._filePath.stem

    def getIndex(self) -> int:
        return self._index

    def getState(self) -> DiffractionPatternState:
        return self._state

    def getData(self) -> DiffractionPatternData:
        self._state = DiffractionPatternState.MISSING

        with TiffFile(self._filePath) as tiff:
            try:
                data = tiff.asarray()
            except:
                raise
            else:
                self._state = DiffractionPatternState.FOUND

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        return data


class TiffDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'TIFF'

    @property
    def fileFilter(self) -> str:
        return 'Tagged Image File Format Files (*.tif *.tiff)'

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(0, 0, numpy.dtype(numpy.ubyte), filePath=filePath)
        arrayList: list[DiffractionPatternArray] = list()

        if filePath:
            digits = re.findall(r'\d+', filePath.stem)
            longest_digits = max(digits, key=len)
            pattern = filePath.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')
            filePathList: list[Path] = list()

            for fp in filePath.parent.iterdir():
                z = re.match(pattern, fp.name)

                if z:
                    index = int(z.group(1).lstrip('0'))
                    filePathList.append(fp)

            filePathList.sort(key=lambda fp: fp.stem)

            for fp in filePathList:
                array: DiffractionPatternArray = TiffDiffractionPatternArray(fp, len(arrayList))
                arrayList.append(array)

            with TiffFile(filePath) as tiff:
                data = tiff.asarray()
                metadata = DiffractionMetadata(
                    filePath=filePath.parent / pattern,
                    numberOfPatternsPerArray=len(tiff.pages),
                    numberOfPatternsTotal=len(tiff.pages) * len(arrayList),
                    patternDataType=data.dtype,
                )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

        for array in arrayList:
            contentsTree.createChild([array.getLabel(), 'TIFF', str(array.getIndex())])

        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(TiffDiffractionFileReader())


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiffFile = reader.read(filePath)
    print(tiffFile)
