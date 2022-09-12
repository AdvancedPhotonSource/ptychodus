from __future__ import annotations
import logging
import queue
import tempfile
import threading

import numpy

from ...api.data import DiffractionData
from ...api.observer import Observable
from .crop import CropSizer
from .settings import DataSettings

logger = logging.getLogger(__name__)


class DiffractionDataAssembler(Observable):

    def __init__(self, settings: DataSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._cropSizer = cropSizer
        self._dataQueue: queue.Queue[DiffractionData] = queue.Queue()
        self._dataArray = numpy.empty((0, 0, 0), dtype=numpy.uint16)
        self._consumerThreads: list[threading.Thread] = list()
        self._consumerStopEvent = threading.Event()
        self._changedEvent = threading.Event()

    @property
    def isActive(self) -> bool:
        return len(self._consumerThreads) > 0

    def _clearQueue(self) -> None:
        with self._dataQueue.mutex:
            self._dataQueue.queue.clear()

    def _reallocateDataArray(self, maximumNumberOfImages: int) -> None:
        shape = (
            maximumNumberOfImages,
            self._cropSizer.getExtentYInPixels(),
            self._cropSizer.getExtentXInPixels(),
        )
        scratchDirectory = self._settings.scratchDirectory.value

        if scratchDirectory.is_dir():
            npyTempFile = tempfile.NamedTemporaryFile(dir=scratchDirectory, suffix='.npy')
            logger.debug(f'Scratch data file {npyTempFile.name} is {shape}')
            self._dataArray = numpy.memmap(npyTempFile, dtype=numpy.uint16, shape=shape)
        else:
            logger.debug(f'Scratch memory is {shape}')
            self._dataArray = numpy.zeros(shape, dtype=numpy.uint16)

    def _startConsumerThreads(self) -> None:
        self._consumerStopEvent.clear()

        for idx in range(self._settings.numberOfDataThreads.value):
            thread = threading.Thread(target=self._assemble)
            thread.start()
            self._consumerThreads.append(thread)

    def start(self, maximumNumberOfImages: int) -> None:
        if self.isActive:
            self.stop()

        logger.info('Starting data assembler...')
        self._reallocateDataArray(maximumNumberOfImages)
        self._startConsumerThreads()
        logger.info('Data assembler started.')

    def _stopConsumerThreads(self) -> None:
        self._consumerStopEvent.set()

        for thread in self._consumerThreads:
            thread.join()

        self._consumerThreads.clear()

    def stop(self) -> None:
        logger.info('Stopping data assembler...')
        self._stopConsumerThreads()
        self._clearQueue()
        logger.info('Data assembler stopped.')

    def _assemble(self) -> None:
        # FIXME maintain scan order; maybe bool array?
        while not self._consumerStopEvent.is_set():
            try:
                data = self._dataQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            logger.debug(f'Reading {data.name}...')
            array = data.getArray()

            startIndex = data.getStartIndex()
            sliceZ = slice(startIndex, startIndex + array.shape[0])

            if self._cropSizer.isCropEnabled():
                sliceY = self._cropSizer.getSliceY()
                sliceX = self._cropSizer.getSliceX()
                array = array[:, sliceY, sliceX]

            threshold = self._settings.threshold.value
            array[array < threshold] = threshold

            if self._settings.flipX.value:
                array = numpy.fliplr(array)

            if self._settings.flipY.value:
                array = numpy.flipud(array)

            self._dataArray[sliceZ, :, :] = array
            self._changedEvent.set()

    def assemble(self, data: DiffractionData) -> None:
        self._dataQueue.put(data)

    def processEvents(self) -> None:
        # FIXME let main thread trigger gui updates on timer
        if self._changedEvent.is_set():
            self.notifyObservers()
            self._changedEvent.clear()
