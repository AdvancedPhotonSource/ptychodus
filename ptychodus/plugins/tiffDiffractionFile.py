from pathlib import Path
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.data import (DiffractionArrayType, DiffractionData, DiffractionDataState,
                                DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                                DiffractionMetadata, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionData(DiffractionData):

    def __init__(self, filePath: Path) -> None:
        super().__init__()
        self._filePath = filePath
        self._state = DiffractionDataState.MISSING

    @property
    def name(self) -> str:
        return self._filePath.stem

    def getState(self) -> DiffractionDataState:
        return self._state

    def getStartIndex(self) -> int:
        return 0

    def getArray(self) -> DiffractionArrayType:
        array = numpy.empty((0, 0, 0), dtype=numpy.uint16)

        if self._filePath.is_file():
            self._state = DiffractionDataState.FOUND

            try:
                with TiffFile(self._filePath) as tif:
                    array = tif.asarray()
            except OSError as err:
                logger.exception(err)
            else:
                self._state = DiffractionDataState.LOADED

                if array.ndim == 2:
                    array = array[numpy.newaxis, :, :]
        else:
            self._state = DiffractionDataState.MISSING

        return array


class TiffDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'TIFF'

    @property
    def fileFilter(self) -> str:
        return 'Tagged Image File Format Files (*.tif *.tiff)'

    def read(self, filePath: Path) -> DiffractionDataset:
        metadata = DiffractionMetadata(filePath, 0, 0, 0)
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        dataList: list[DiffractionData] = list()

        if filePath:
            digits = re.findall(r'\d+', filePath.stem)
            longest_digits = max(digits, key=len)
            pattern = filePath.name.replace(longest_digits, f'\\d{{{len(longest_digits)}}}')
            totalNumberOfImages = 0

            for fp in filePath.parent.iterdir():
                if re.search(pattern, fp.name):
                    with TiffFile(fp) as tiff:
                        totalNumberOfImages += len(tiff.pages)

                        itemName = fp.stem
                        itemType = 'TIFF'
                        itemDetails = str(tiff.epics_metadata)
                        contentsTree.createChild([itemName, itemType, itemDetails])

                    dataset = TiffDiffractionData(fp)
                    dataList.append(dataset)

            with TiffFile(filePath) as tiff:
                array = tiff.asarray()
                imageWidth = array.shape[-1]
                imageHeight = array.shape[-2]

            metadata = DiffractionMetadata(filePath.parent / pattern, imageWidth, imageHeight,
                                           totalNumberOfImages)

        return SimpleDiffractionDataset(metadata, contentsTree, dataList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(TiffDiffractionFileReader())


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiffFile = reader.read(filePath)
    print(tiffFile)
