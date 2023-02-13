from __future__ import annotations
import logging
import queue
import threading

import numpy

from ...api.data import (DiffractionDataset, DiffractionPatternArray, DiffractionPatternState,
                         SimpleDiffractionPatternArray)
from .dataset import ActiveDiffractionDataset
from .settings import DiffractionDatasetSettings

__all__ = [
    'ActiveDiffractionDatasetBuilder',
]

logger = logging.getLogger(__name__)


class ActiveDiffractionDatasetBuilder:

    def __init__(self, settings: DiffractionDatasetSettings,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
        self._arrayQueue: queue.Queue[DiffractionPatternArray] = queue.Queue()
        self._workers: list[threading.Thread] = list()
        self._stopWorkEvent = threading.Event()

    @property
    def isAssembling(self) -> bool:
        return (len(self._workers) > 0)

    def _getArrayAndAssemble(self) -> None:
        while not self._stopWorkEvent.is_set():
            try:
                array = self._arrayQueue.get(block=True, timeout=1)

                try:
                    self._assemble(array)
                finally:
                    self._arrayQueue.task_done()
            except queue.Empty:
                pass
            except:
                logger.exception('Error while assembling array!')

    def _assemble(self, array: DiffractionPatternArray) -> None:
        logger.debug(f'Assembling {array.getLabel()}...')

        try:
            data = array.getData()
        except:
            data = numpy.zeros((0, 0, 0), dtype=numpy.uint16)
            state = DiffractionPatternState.MISSING
        else:
            state = DiffractionPatternState.LOADED

        array = SimpleDiffractionPatternArray(array.getLabel(), array.getIndex(), data, state)

        self._dataset.insertArray(array)

    def insertArray(self, array: DiffractionPatternArray) -> None:
        self._arrayQueue.put(array)

    def switchTo(self, dataset: DiffractionDataset) -> None:
        if self.isAssembling:
            self.stop(finishAssembling=False)

        self._dataset.reset(dataset.getMetadata(), dataset.getContentsTree())

        for array in dataset:
            self.insertArray(array)

    def start(self) -> None:
        if self.isAssembling:
            self.stop(finishAssembling=False)

        logger.info('Starting data assembler...')
        self._stopWorkEvent.clear()

        for idx in range(self._settings.numberOfDataThreads.value):
            thread = threading.Thread(target=self._getArrayAndAssemble)
            thread.start()
            self._workers.append(thread)

        logger.info('Data assembler started.')

    def stop(self, finishAssembling: bool) -> None:
        if finishAssembling:
            self._arrayQueue.join()

        logger.info('Stopping data assembler...')
        self._stopWorkEvent.set()

        while self._workers:
            thread = self._workers.pop()
            thread.join()

        with self._arrayQueue.mutex:
            self._arrayQueue.queue.clear()

        logger.info('Data assembler stopped.')

    def getAssemblyQueueSize(self) -> int:
        return self._arrayQueue.qsize()
