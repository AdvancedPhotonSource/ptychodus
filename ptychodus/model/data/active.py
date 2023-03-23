from __future__ import annotations
from collections.abc import Sequence
from typing import overload, Any, Union
import logging
import tempfile
import threading

import numpy
import numpy.typing

from ...api.data import (DiffractionDataset, DiffractionMetadata, DiffractionPatternArray,
                         DiffractionPatternData, DiffractionPatternState,
                         SimpleDiffractionPatternArray)
from ...api.geometry import Array2D
from ...api.tree import SimpleTreeNode
from .settings import DiffractionDatasetSettings, DiffractionPatternSettings
from .sizer import DiffractionPatternSizer

__all__ = [
    'ActiveDiffractionDataset',
]

DiffractionPatternIndexes = numpy.typing.NDArray[numpy.integer[Any]]

logger = logging.getLogger(__name__)


class ActiveDiffractionDataset(DiffractionDataset):

    def __init__(self, datasetSettings: DiffractionDatasetSettings,
                 patternSettings: DiffractionPatternSettings,
                 diffractionPatternSizer: DiffractionPatternSizer) -> None:
        super().__init__()
        self._datasetSettings = datasetSettings
        self._patternSettings = patternSettings
        self._diffractionPatternSizer = diffractionPatternSizer

        self._metadata = DiffractionMetadata.createNullInstance()
        self._contentsTree = SimpleTreeNode.createRoot(list())
        self._arrayListLock = threading.RLock()
        self._arrayList: list[DiffractionPatternArray] = list()
        self._arrayData: DiffractionPatternData = numpy.zeros((1, 1, 1), dtype=numpy.uint16)
        self._changedEvent = threading.Event()

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> \
            Union[DiffractionPatternArray, Sequence[DiffractionPatternArray]]:
        with self._arrayListLock:
            return self._arrayList[index]

    def __len__(self) -> int:
        with self._arrayListLock:
            return len(self._arrayList)

    def reset(self, metadata: DiffractionMetadata, contentsTree: SimpleTreeNode) -> None:
        with self._arrayListLock:
            self._metadata = metadata
            self._contentsTree = contentsTree
            self._arrayList.clear()

        self._changedEvent.set()

    def realloc(self) -> None:
        shape = (
            self._metadata.numberOfPatternsTotal,
            self._diffractionPatternSizer.getExtentYInPixels(),
            self._diffractionPatternSizer.getExtentXInPixels(),
        )

        with self._arrayListLock:
            self._arrayList.clear()

            if self._datasetSettings.memmapEnabled.value:
                scratchDirectory = self._datasetSettings.scratchDirectory.value
                scratchDirectory.mkdir(mode=0o755, parents=True, exist_ok=True)
                npyTempFile = tempfile.NamedTemporaryFile(dir=scratchDirectory, suffix='.npy')
                logger.debug(f'Scratch data file {npyTempFile.name} is {shape}')
                self._arrayData = numpy.memmap(npyTempFile,
                                               dtype=self._metadata.patternDataType,
                                               shape=shape)
                self._arrayData[:] = 0
            else:
                logger.debug(f'Scratch memory is {shape}')
                self._arrayData = numpy.zeros(shape, dtype=self._metadata.patternDataType)

        self._changedEvent.set()

    def insertArray(self, array: DiffractionPatternArray) -> None:
        if array.getState() == DiffractionPatternState.LOADED:
            data = self._diffractionPatternSizer(array.getData())

            if self._patternSettings.thresholdEnabled.value:
                thresholdValue = self._patternSettings.thresholdValue.value
                data[data < thresholdValue] = thresholdValue

            if self._patternSettings.flipXEnabled.value:
                data = numpy.flip(data, axis=-1)

            if self._patternSettings.flipYEnabled.value:
                data = numpy.flip(data, axis=-2)

            offset = self._metadata.numberOfPatternsPerArray * array.getIndex()
            sliceZ = slice(offset, offset + data.shape[0])
            dataView = self._arrayData[sliceZ, :, :]
            dataView[:] = data
            dataView.flags.writeable = False

            array = SimpleDiffractionPatternArray(array.getLabel(), array.getIndex(), dataView,
                                                  array.getState())

        with self._arrayListLock:
            self._arrayList.append(array)
            self._arrayList.sort(key=lambda arr: arr.getIndex())

        self._changedEvent.set()

    def getAssembledIndexes(self) -> list[int]:
        indexes: list[int] = list()

        with self._arrayListLock:
            for array in self._arrayList:
                if array.getState() == DiffractionPatternState.LOADED:
                    offset = self._metadata.numberOfPatternsPerArray * array.getIndex()
                    size = array.getNumberOfPatterns()
                    indexes.extend(range(offset, offset + size))

        return indexes

    def getAssembledData(self) -> DiffractionPatternData:
        indexes = self.getAssembledIndexes()
        return self._arrayData[indexes]

    def setAssembledData(self, arrayData: DiffractionPatternData,
                         arrayIndexes: DiffractionPatternIndexes) -> None:
        with self._arrayListLock:
            numberOfPatterns, detectorHeight, detectorWidth = arrayData.shape

            self._metadata = DiffractionMetadata(
                numberOfPatternsPerArray=numberOfPatterns,
                numberOfPatternsTotal=numberOfPatterns,
                patternDataType=arrayData.dtype,
                detectorNumberOfPixels=Array2D[int](detectorWidth, detectorHeight),
            )

            self._contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

            # TODO use arrayIndexes
            self._arrayList = [
                SimpleDiffractionPatternArray(
                    label='Restart',
                    index=0,
                    data=arrayData[...],
                    state=DiffractionPatternState.LOADED,
                ),
            ]
            self._arrayData = arrayData

        self.notifyObservers()

    def notifyObserversIfDatasetChanged(self) -> None:
        if self._changedEvent.is_set():
            self._changedEvent.clear()
            self.notifyObservers()
