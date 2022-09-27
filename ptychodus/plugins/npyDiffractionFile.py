from pathlib import Path
import logging

import numpy

from ptychodus.api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                                DiffractionPatternArray, DiffractionPatternData,
                                DiffractionPatternState, SimpleDiffractionDataset)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class NPYDiffractionPatternArray(DiffractionPatternArray):

    def __init__(self, filePath: Path) -> None:
        super().__init__()
        self._filePath = filePath
        self._state = DiffractionPatternState.UNKNOWN

    def getLabel(self) -> str:
        return self._filePath.stem

    def getIndex(self) -> int:
        return 0

    def getState(self) -> DiffractionPatternState:
        return self._state

    def getData(self) -> DiffractionPatternData:
        try:
            data = numpy.load(self._filePath)
        except:
            self._state = DiffractionPatternState.MISSING
            raise
        else:
            self._state = DiffractionPatternState.FOUND

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        return data


class NPYDiffractionFileReader(DiffractionFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> DiffractionDataset:
        array = NPYDiffractionPatternArray(filePath)
        data = array.getData()
        metadata = DiffractionMetadata(
            filePath=filePath,
            numberOfPatternsPerArray=data.shape[0],
            numberOfPatternsTotal=data.shape[0],
        )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = [array]
        return SimpleDiffractionDataset(metadata, contentsTree, arrayList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPYDiffractionFileReader())
