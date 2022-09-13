from pathlib import Path
import logging

import numpy

from ptychodus.api.data import (DiffractionArrayType, DiffractionData, DiffractionDataState,
                                DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                                DiffractionMetadata, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPYDiffractionData(DiffractionData):

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
                array = numpy.load(self._filePath)
            except OSError as err:
                logger.exception(err)
            else:
                self._state = DiffractionDataState.LOADED

                if array.ndim == 2:
                    array = array[numpy.newaxis, :, :]
        else:
            self._state = DiffractionDataState.MISSING

        return array


class NPYDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> DiffractionDataset:
        data = NPYDiffractionData(filePath)
        array = data.getArray()
        metadata = DiffractionMetadata(filePath, array.shape[1], array.shape[2], array.shape[0])

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        dataList: list[DiffractionData] = [data]
        return SimpleDiffractionDataset(metadata, contentsTree, dataList)


class NPYDiffractionFileWriter(DiffractionFileWriter):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        array = dataset[0].getArray()
        numpy.save(filePath, array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYDiffractionFileReader())
    registry.registerPlugin(NPYDiffractionFileWriter())
