from collections.abc import Sequence
from pathlib import Path
from typing import overload, Union
import logging
import re
import sys

from tifffile import TiffFile
import numpy

from ptychodus.api.data import DataArrayType, DataFile, DataFileMetadata, DataFileReader, \
        DatasetState, DiffractionDataset
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class TiffDiffractionDataset(DiffractionDataset):

    def __init__(self, filePath: Path) -> None:
        super().__init__()
        self._filePath = filePath
        self._state = DatasetState.NOT_FOUND

    @property
    def datasetName(self) -> str:
        return self._filePath.stem

    @property
    def datasetState(self) -> DatasetState:
        return self._state

    @overload
    def __getitem__(self, index: int) -> DataArrayType:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DataArrayType]:
        ...

    def __getitem__(self, index: Union[int,
                                       slice]) -> Union[DataArrayType, Sequence[DataArrayType]]:
        array = self.getArray()
        return array[index, ...]

    def __len__(self) -> int:
        array = self.getArray()
        return array.shape[0]

    def getArray(self) -> DataArrayType:
        array = numpy.empty((0, 0, 0), dtype=numpy.uint16)

        if self._filePath.is_file():
            self._state = DatasetState.EXISTS

            try:
                with TiffFile(self._filePath) as tif:
                    array = tif.asarray()

                    if array.ndim == 2:
                        array = array[numpy.newaxis, :, :]
            except OSError as err:
                logger.exception(err)
            else:
                self._state = DatasetState.VALID
        else:
            self._state = DatasetState.NOT_FOUND

        return array


class TiffDataFile(DataFile):

    def __init__(self, metadata: DataFileMetadata, contentsTree: SimpleTreeNode,
                 datasetList: list[DiffractionDataset]) -> None:
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._datasetList = datasetList

    @property
    def metadata(self) -> DataFileMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionDataset:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionDataset]:
        ...

    def __getitem__(
            self, index: Union[int,
                               slice]) -> Union[DiffractionDataset, Sequence[DiffractionDataset]]:
        return self._datasetList[index]

    def __len__(self) -> int:
        return len(self._datasetList)


class TiffDataFileReader(DataFileReader):

    @property
    def simpleName(self) -> str:
        return 'TIFF'

    @property
    def fileFilter(self) -> str:
        return 'Tagged Image File Format Files (*.tif *.tiff)'

    def read(self, filePath: Path) -> DataFile:
        metadata = DataFileMetadata(filePath, 0, 0, 0)
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        datasetList: list[DiffractionDataset] = list()

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

                    dataset = TiffDiffractionDataset(fp)
                    datasetList.append(dataset)

            with TiffFile(filePath) as tiff:
                array = tiff.asarray()
                imageWidth = array.shape[-1]
                imageHeight = array.shape[-2]

            metadata = DataFileMetadata(filePath.parent / pattern, imageWidth, imageHeight,
                                        totalNumberOfImages)

        return TiffDataFile(metadata, contentsTree, datasetList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(TiffDataFileReader())


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    reader = TiffDataFileReader()
    tiffFile = reader.read(filePath)
    print(tiffFile)
