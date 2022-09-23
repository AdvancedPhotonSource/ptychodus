from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Union
import logging
import queue
import tempfile
import threading

import numpy

from ...api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                         DiffractionDataset, DiffractionMetadata, SimpleDiffractionArray)
from ...api.observer import Observable, Observer
from ...api.tree import SimpleTreeNode
from .crop import CropSizer
from .settings import DataSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyTask:
    array: DiffractionArray
    offset: int


class ActiveDiffractionDataset(DiffractionDataset):

    def __init__(self, settings: DataSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._cropSizer = cropSizer

        self._metadata = DiffractionMetadata.createNullInstance()
        self._contentsTree = SimpleTreeNode.createRoot(list())
        self._arrayDict: dict[int, DiffractionArray] = dict()
        self._arrayDictLock = threading.RLock()
        self._arrayData: DiffractionDataType = numpy.zeros(
            (1, 1, 1), dtype=self._settings.arrayDataType.value)

        self._taskQueue: queue.Queue[AssemblyTask] = queue.Queue()
        self._workers: list[threading.Thread] = list()
        self._stopWorkEvent = threading.Event()
        self._changedEvent = threading.Event()

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]:
        ...

    def __getitem__(
            self, index: Union[int, slice]) -> Union[DiffractionArray, Sequence[DiffractionArray]]:
        if isinstance(index, slice):
            valueList: list[DiffractionArray] = list()

            with self._arrayDictLock:
                for idx in range(index.start, index.stop, index.step):
                    value = self._arrayDict.get(idx)

                    if value:
                        valueList.append(value)

            return valueList
        else:
            with self._arrayDictLock:
                return self._arrayDict[index]

    def __len__(self) -> int:
        with self._arrayDictLock:
            return max(self._arrayDict.keys(), default=0)

    def _reallocate(self, maximumNumberOfImages: int) -> None:
        scratchDirectory = self._settings.scratchDirectory.value
        dtype = self._settings.arrayDataType.value
        shape = (
            maximumNumberOfImages,
            self._cropSizer.getExtentYInPixels(),
            self._cropSizer.getExtentXInPixels(),
        )

        if scratchDirectory.is_dir():
            npyTempFile = tempfile.NamedTemporaryFile(dir=scratchDirectory, suffix='.npy')
            logger.debug(f'Scratch data file {npyTempFile.name} is {shape}')
            self._arrayData = numpy.memmap(npyTempFile, dtype=dtype, shape=shape)
            self._arrayData[:] = 0
        else:
            logger.debug(f'Scratch memory is {shape}')
            self._arrayData = numpy.zeros(shape, dtype=dtype)

        with self._arrayDictLock:
            self._arrayDict.clear()

    def _assemble(self) -> None:
        while not self._stopWorkEvent.is_set():
            try:
                task = self._taskQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            logger.debug(f'Reading {task.array.getLabel()}...')
            data = task.array.getData()

            if self._cropSizer.isCropEnabled():
                sliceY = self._cropSizer.getSliceY()
                sliceX = self._cropSizer.getSliceX()
                data = data[:, sliceY, sliceX]

            threshold = self._settings.threshold.value
            data[data < threshold] = threshold

            if self._settings.flipX.value:
                data = numpy.fliplr(data)

            if self._settings.flipY.value:
                data = numpy.flipud(data)

            sliceZ = slice(task.offset, task.offset + data.shape[0])
            dataView = self._arrayData[sliceZ, :, :]
            dataView = data
            dataView.flags.writeable = False

            arrayView = SimpleDiffractionArray(
                task.array.getLabel(),
                task.array.getIndex(),
                dataView,
                DiffractionArrayState.LOADED,
            )

            self._changedEvent.set()

    @property
    def isActive(self) -> bool:
        return (len(self._workers) > 0)

    def start(self, maximumNumberOfImages: int) -> None:
        if self.isActive:
            self.stop()

        logger.info('Starting data assembler...')
        self._stopWorkEvent.clear()

        self._reallocate(maximumNumberOfImages)

        for idx in range(self._settings.numberOfDataThreads.value):
            thread = threading.Thread(target=self._assemble)
            thread.start()
            self._workers.append(thread)

        logger.info('Data assembler started.')

    def stop(self) -> None:
        logger.info('Stopping data assembler...')
        self._stopWorkEvent.set()

        while self._workers:
            thread = self._workers.pop()
            thread.join()

        with self._taskQueue.mutex:
            self._taskQueue.queue.clear()

        logger.info('Data assembler stopped.')

    def switchTo(self, dataset: DiffractionDataset) -> None:
        if self.isActive:
            self.stop()

        self._metadata = dataset.getMetadata()
        self._contentsTree = dataset.getContentsTree()
        self.start(self._metadata.numberOfImagesTotal)

        for array in dataset:
            self.insertArray(array)

        self.notifyObservers()

    def insertArray(self, array: DiffractionArray) -> None:
        offset = self._metadata.numberOfImagesPerArray * array.getIndex()
        task = AssemblyTask(array, offset)
        self._taskQueue.put(task)

    def getAssembledData(self) -> DiffractionDataType:
        indexes = list(self._arrayDict.keys())
        indexes.sort()
        return self._arrayData[indexes]

    def notifyObserversIfDatasetChanged(self) -> None:
        # FIXME let main thread trigger gui updates on timer
        if self._changedEvent.is_set():
            self.notifyObservers()
            self._changedEvent.clear()
