from __future__ import annotations
from collections.abc import Sequence
from typing import overload
import logging
import tempfile
import threading

import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionMetadata,
    DiffractionPatternArray,
    DiffractionPatternArrayType,
    DiffractionPatternIndexes,
    DiffractionPatternState,
    SimpleDiffractionPatternArray,
)
from ptychodus.api.tree import SimpleTreeNode

from .settings import PatternSettings
from .sizer import PatternSizer

__all__ = [
    "ActiveDiffractionDataset",
]

logger = logging.getLogger(__name__)


class ActiveDiffractionDataset(DiffractionDataset):

    def __init__(self, settings: PatternSettings, diffractionPatternSizer: PatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._diffractionPatternSizer = diffractionPatternSizer

        self._metadata = DiffractionMetadata.createNullInstance()
        self._contentsTree = SimpleTreeNode.createRoot(list())
        self._arrayListLock = threading.RLock()
        self._arrayList: list[DiffractionPatternArray] = list()
        self._arrayData: DiffractionPatternArrayType = numpy.zeros((0, 0, 0), dtype=numpy.uint16)
        self._changedEvent = threading.Event()

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    def getInfoText(self) -> str:
        filePath = self._metadata.filePath
        label = filePath.stem if filePath else "None"
        number, height, width = self._arrayData.shape
        dtype = str(self._arrayData.dtype)
        sizeInMB = self._arrayData.nbytes / (1024 * 1024)
        return f"{label}: {number} x {width}W x {height}H {dtype} [{sizeInMB:.2f}MB]"

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]:
        ...

    def __getitem__(
            self,
            index: int | slice) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
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
            self._diffractionPatternSizer.getHeightInPixels(),
            self._diffractionPatternSizer.getWidthInPixels(),
        )

        with self._arrayListLock:
            self._arrayList.clear()

            if self._settings.memmapEnabled.getValue():
                scratchDirectory = self._settings.scratchDirectory.getValue()
                scratchDirectory.mkdir(mode=0o755, parents=True, exist_ok=True)
                npyTempFile = tempfile.NamedTemporaryFile(dir=scratchDirectory, suffix=".npy")
                logger.debug(f"Scratch data file {npyTempFile.name} is {shape}")
                self._arrayData = numpy.memmap(npyTempFile,
                                               dtype=self._metadata.patternDataType,
                                               shape=shape)
                self._arrayData[:] = 0
            else:
                logger.debug(f"Scratch memory is {shape}")
                self._arrayData = numpy.zeros(shape, dtype=self._metadata.patternDataType)

        self._changedEvent.set()

    def insertArray(self, array: DiffractionPatternArray) -> None:
        if array.getState() == DiffractionPatternState.LOADED:
            data = self._diffractionPatternSizer(array.getData())

            if self._settings.valueUpperBoundEnabled.getValue():
                valueLowerBound = self._settings.valueLowerBound.getValue()
                valueUpperBound = self._settings.valueUpperBound.getValue()
                data[data >= valueUpperBound] = 0

            if self._settings.valueLowerBoundEnabled.getValue():
                valueLowerBound = self._settings.valueLowerBound.getValue()
                data[data < valueLowerBound] = 0

            if self._settings.flipXEnabled.getValue():
                data = numpy.flip(data, axis=-1)

            if self._settings.flipYEnabled.getValue():
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

    def getAssembledIndexes(self) -> Sequence[int]:
        indexes: list[int] = list()

        with self._arrayListLock:
            for array in self._arrayList:
                if array.getState() == DiffractionPatternState.LOADED:
                    offset = self._metadata.numberOfPatternsPerArray * array.getIndex()
                    size = array.getNumberOfPatterns()
                    indexes.extend(range(offset, offset + size))

        return indexes

    def getAssembledData(self) -> DiffractionPatternArrayType:
        indexes = self.getAssembledIndexes()
        return self._arrayData[indexes]

    def setAssembledData(
        self,
        arrayData: DiffractionPatternArrayType,
        arrayIndexes: DiffractionPatternIndexes,
    ) -> None:
        with self._arrayListLock:
            numberOfPatterns, detectorHeight, detectorWidth = arrayData.shape

            self._metadata = DiffractionMetadata(
                numberOfPatternsPerArray=numberOfPatterns,
                numberOfPatternsTotal=numberOfPatterns,
                patternDataType=arrayData.dtype,
                detectorExtent=ImageExtent(detectorWidth, detectorHeight),
            )

            self._contentsTree = SimpleTreeNode.createRoot(["Name", "Type", "Details"])

            # TODO use arrayIndexes
            self._arrayList = [
                SimpleDiffractionPatternArray(
                    label="Processed",
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
