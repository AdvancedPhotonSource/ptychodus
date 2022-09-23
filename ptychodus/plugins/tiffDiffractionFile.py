from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                                DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                                DiffractionMetadata, SimpleDiffractionArray,
                                SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionArray(DiffractionArray):

    def __init__(self, filePath: Path, index: int) -> None:
        super().__init__()
        self._filePath = filePath
        self._index = index
        self._state = DiffractionArrayState.UNKNOWN

    def getLabel(self) -> str:
        return self._filePath.stem

    def getIndex(self) -> int:
        return self._index

    def getState(self) -> DiffractionArrayState:
        return self._state

    def getData(self) -> DiffractionDataType:
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)

        if self._filePath.is_file():
            self._state = DiffractionArrayState.FOUND

            try:
                with TiffFile(self._filePath) as tiff:
                    data = tiff.asarray()
            except OSError as err:
                logger.debug(f'Unable to read \"{self.getLabel()}\"!')
            else:
                self._state = DiffractionArrayState.LOADED

                if data.ndim == 2:
                    data = data[numpy.newaxis, :, :]
        else:
            self._state = DiffractionArrayState.MISSING

        return data


class TiffDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'TIFF'

    @property
    def fileFilter(self) -> str:
        return 'Tagged Image File Format Files (*.tif *.tiff)'

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(filePath, 0, 0, 0, 0)
        arrayList: list[DiffractionArray] = list()

        if filePath:
            digits = re.findall(r'\d+', filePath.stem)
            longest_digits = max(digits, key=len)
            pattern = filePath.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

            for fp in filePath.parent.iterdir():
                z = re.match(pattern, fp.name)

                if z:
                    index = int(z.group(1).lstrip('0'))
                    array: DiffractionArray = TiffDiffractionArray(fp, index)
                    arrayList.append(array)

            with TiffFile(filePath) as tiff:
                data = tiff.asarray()
                metadata = DiffractionMetadata(
                    filePath=filePath.parent / pattern,
                    imageWidth=data.shape[-1],
                    imageHeight=data.shape[-2],
                    numberOfImagesPerArray=len(tiff.pages),
                    numberOfImagesTotal=len(tiff.pages) * len(arrayList),
                )

        arrayList.sort(key=lambda array: array.getIndex())
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
