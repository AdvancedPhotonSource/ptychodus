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

from ...api.data import (DiffractionDataset, DiffractionMetadata, DiffractionPatternArray,
                         DiffractionPatternData, DiffractionPatternState, SimpleDiffractionDataset,
                         SimpleDiffractionPatternArray)
from ...api.observer import Observable, Observer
from ...api.tree import SimpleTreeNode
from .crop import CropSizer
from .settings import DiffractionDatasetSettings, DiffractionPatternSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyTask:
    array: DiffractionPatternArray
    offset: int


class ActiveDiffractionDataset(DiffractionDataset):

    def __init__(self, datasetSettings: DiffractionDatasetSettings,
                 patternSettings: DiffractionPatternSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._datasetSettings = datasetSettings
        self._patternSettings = patternSettings
        self._cropSizer = cropSizer

        self._dataset: DiffractionDataset = SimpleDiffractionDataset.createNullInstance()
        self._arrayList: list[DiffractionPatternArray] = list()
        self._arrayListLock = threading.RLock()
        self._arrayData: DiffractionPatternData = numpy.zeros((1, 1, 1), dtype=numpy.uint16)

        self._taskQueue: queue.Queue[AssemblyTask] = queue.Queue()
        self._workers: list[threading.Thread] = list()
        self._stopWorkEvent = threading.Event()
        self._changedEvent = threading.Event()

    @property
    def isReadyToAssemble(self) -> bool:
        return (len(self) < len(self._dataset))

    @property
    def isAssembling(self) -> bool:
        return (len(self._workers) > 0)

    def getMetadata(self) -> DiffractionMetadata:
        return self._dataset.getMetadata()

    def getContentsTree(self) -> SimpleTreeNode:
        return self._dataset.getContentsTree()

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

    def _getTaskAndAssemble(self) -> None:
        while not self._stopWorkEvent.is_set():
            try:
                task = self._taskQueue.get(block=True, timeout=1)

                try:
                    self._assemble(task)
                finally:
                    self._taskQueue.task_done()
            except queue.Empty:
                pass
            except:
                logger.exception('Error while assembling array!')

    def _assemble(self, task: AssemblyTask) -> None:
        logger.debug(f'Assembling {task.array.getLabel()}...')

        try:
            data = task.array.getData()
        except:
            sliceZ = slice(task.offset, task.offset + self.getMetadata().numberOfPatternsPerArray)
            dataView = self._arrayData[sliceZ, :, :]
            dataView.flags.writeable = False

            arrayView = SimpleDiffractionPatternArray(
                task.array.getLabel(),
                task.array.getIndex(),
                dataView,
                DiffractionPatternState.MISSING,
            )
        else:
            if self._cropSizer.isCropEnabled():
                sliceY = self._cropSizer.getSliceY()
                sliceX = self._cropSizer.getSliceX()
                data = data[:, sliceY, sliceX]

            threshold = self._patternSettings.threshold.value
            data[data < threshold] = threshold

            if self._patternSettings.flipXEnabled.value:
                data = numpy.fliplr(data)

            if self._patternSettings.flipYEnabled.value:
                data = numpy.flipud(data)

            sliceZ = slice(task.offset, task.offset + data.shape[0])
            dataView = self._arrayData[sliceZ, :, :]
            dataView[:] = data
            dataView.flags.writeable = False

            arrayView = SimpleDiffractionPatternArray(
                task.array.getLabel(),
                task.array.getIndex(),
                dataView,
                DiffractionPatternState.LOADED,
            )

        with self._arrayListLock:
            self._arrayList.append(arrayView)
            self._arrayList.sort(key=lambda array: array.getIndex())

        self._changedEvent.set()

    def switchTo(self, dataset: DiffractionDataset) -> None:
        if self.isAssembling:
            self.stop()

        self._arrayList.clear()
        self._dataset = dataset
        self.notifyObservers()

    def start(self, block: bool) -> None:
        if self.isAssembling:
            self.stop()

        logger.info('Resetting data assembler...')

        scratchDirectory = self._datasetSettings.scratchDirectory.value
        maximumNumberOfPatterns = self._dataset.getMetadata().numberOfPatternsTotal
        dtype = self._dataset.getMetadata().patternDataType
        shape = (
            maximumNumberOfPatterns,
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

        self._arrayList.clear()

        for array in self._dataset:
            self.insertArray(array)

        logger.info('Data assembler reset.')
        logger.info('Starting data assembler...')
        self._stopWorkEvent.clear()

        for idx in range(self._patternSettings.numberOfDataThreads.value):
            thread = threading.Thread(target=self._getTaskAndAssemble)
            thread.start()
            self._workers.append(thread)

        logger.info('Data assembler started.')

        if block:
            self._taskQueue.join()
            self.stop()

    def insertArray(self, array: DiffractionPatternArray) -> None:
        stride = self.getMetadata().numberOfPatternsPerArray
        offset = stride * array.getIndex()
        task = AssemblyTask(array, offset)
        self._taskQueue.put(task)

    def stop(self) -> None:
        logger.info('Stopping data assembler...')
        self._stopWorkEvent.set()

        while self._workers:
            thread = self._workers.pop()
            thread.join()

        with self._taskQueue.mutex:
            self._taskQueue.queue.clear()

        logger.info('Data assembler stopped.')

    def getAssemblyQueueSize(self) -> int:
        return self._taskQueue.qsize()

    def getAssembledIndexes(self) -> list[int]:
        indexes: list[int] = list()
        stride = self.getMetadata().numberOfPatternsPerArray

        with self._arrayListLock:
            for array in self._arrayList:
                if array.getState() == DiffractionPatternState.LOADED:
                    offset = stride * array.getIndex()
                    size = array.getNumberOfPatterns()
                    indexes.extend(range(offset, offset + size))

        return indexes

    def getAssembledData(self) -> DiffractionPatternData:
        indexes = self.getAssembledIndexes()
        return self._arrayData[indexes]

    def setAssembledData(self, arrayData: DiffractionPatternData) -> None:
        metadata = DiffractionMetadata(
            numberOfPatternsPerArray=arrayData.shape[0],
            numberOfPatternsTotal=arrayData.shape[0],
            patternDataType=arrayData.dtype,
        )

        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])

        arrayList: list[DiffractionPatternArray] = [
            SimpleDiffractionPatternArray(
                label='Restart',
                index=0,
                data=arrayData,
                state=DiffractionPatternState.LOADED,
            ),
        ]

        dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)
        self.switchTo(dataset)
        self.start(block=True)

    def notifyObserversIfDatasetChanged(self) -> None:
        if self._changedEvent.is_set():
            self.notifyObservers()
            self._changedEvent.clear()
