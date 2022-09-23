from pathlib import Path
import logging

import numpy

from ptychodus.api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                                DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                                DiffractionMetadata, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPYDiffractionArray(DiffractionArray):

    def __init__(self, filePath: Path) -> None:
        super().__init__()
        self._filePath = filePath
        self._state = DiffractionArrayState.UNKNOWN

    def getLabel(self) -> str:
        return self._filePath.stem

    def getIndex(self) -> int:
        return 0

    def getState(self) -> DiffractionArrayState:
        return self._state

    def getData(self) -> DiffractionDataType:
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)

        if self._filePath.is_file():
            self._state = DiffractionArrayState.FOUND

            try:
                data = numpy.load(self._filePath)
            except OSError as err:
                logger.debug(f'Unable to read \"{self.getLabel()}\"!')
            else:
                self._state = DiffractionArrayState.LOADED

                if data.ndim == 2:
                    data = data[numpy.newaxis, :, :]
        else:
            self._state = DiffractionArrayState.MISSING

        return data


class NPYDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> DiffractionDataset:
        array = NPYDiffractionArray(filePath)
        data = array.getData()
        metadata = DiffractionMetadata(
            filePath=filePath,
            imageWidth=data.shape[-1],
            imageHeight=data.shape[-2],
            numberOfImagesPerArray=data.shape[0],
            numberOfImagesTotal=data.shape[0],
        )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionArray] = [array]
        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


class NPYDiffractionFileWriter(DiffractionFileWriter):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        data = dataset[0].getData()
        numpy.save(filePath, data)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYDiffractionFileReader())
    registry.registerPlugin(NPYDiffractionFileWriter())
