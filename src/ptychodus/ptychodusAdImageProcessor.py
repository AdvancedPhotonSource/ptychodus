from pathlib import Path
from typing import Any
import logging
import threading
import time

import numpy

from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import pvaccess
import pvapy

from ptychodus.model import ModelCore
import ptychodus


class ReconstructionThread(threading.Thread):
    def __init__(
        self,
        ptychodus: ModelCore,
        inputProductPath: Path,
        outputProductPath: Path,
        reconstructPV: str,
    ) -> None:
        super().__init__()
        self._ptychodus = ptychodus
        self._inputProductPath = inputProductPath
        self._outputProductPath = outputProductPath
        self._channel = pvapy.Channel(reconstructPV, pvapy.CA)
        self._reconstructEvent = threading.Event()
        self._stopEvent = threading.Event()

        self._channel.subscribe('reconstructor', self._monitor)
        self._channel.startMonitor()

    def run(self) -> None:
        while not self._stopEvent.is_set():
            if self._reconstructEvent.wait(timeout=1.0):
                logging.debug('ReconstructionThread: Begin assembling scan positions')
                self._ptychodus_streaming_context.stop()
                logging.debug('ReconstructionThread: End assembling scan positions')
                self._ptychodus.batchModeExecute(
                    'reconstruct', self._inputProductPath, self._outputProductPath
                )
                self._reconstructEvent.clear()
                # reconstruction done; indicate that results are ready
                self._channel.put(0)

    def _monitor(self, pvObject: pvaccess.PvObject) -> None:
        # NOTE caput bdpgp:gp:bit3 1
        logging.debug(f'ReconstructionThread::monitor {pvObject}')

        if pvObject['value']['index'] == 1:
            logging.debug('ReconstructionThread: Reconstruct PV triggered!')
            # start reconstructing
            self._reconstructEvent.set()
        else:
            logging.debug('ReconstructionThread: Reconstruct PV not triggered!')

    def stop(self) -> None:
        self._stopEvent.set()


class PtychodusAdImageProcessor(AdImageProcessor):
    def __init__(self, configDict: dict[str, Any] = {}) -> None:
        super().__init__(configDict)

        self.logger.debug(f'{ptychodus.__name__.title()} ({ptychodus.__version__})')

        settingsFile = configDict.get('settingsFile')
        self._ptychodus = ModelCore(settingsFile)
        self._reconstructionThread = ReconstructionThread(
            self._ptychodus,
            Path(configDict.get('inputProductPath', 'input.npz')),
            Path(configDict.get('outputProductPath', 'output.npz')),
            configDict.get('reconstructPV', 'bdpgp:gp:bit3'),
        )
        self._posXPV = configDict.get('posXPV', 'bluesky:pos_x')
        self._posYPV = configDict.get('posYPV', 'bluesky:pos_y')
        self._nFramesProcessed = 0
        self._processingTime = 0.0

    def start(self) -> None:
        """Called at startup"""
        self._ptychodus.__enter__()
        self._reconstructionThread.start()

    def stop(self) -> None:
        """Called at shutdown"""
        self._reconstructionThread.stop()
        self._reconstructionThread.join()
        self._ptychodus.__exit__(None, None, None)

    def configure(self, configDict: dict[str, Any]) -> None:
        """Configures user processor"""
        numberOfPatternsTotal = configDict['nPatternsTotal']
        numberOfPatternsPerArray = configDict.get('nPatternsPerArray', 1)
        patternDataType = configDict.get('PatternDataType', 'uint16')

        metadata = ptychodus.api.patterns.DiffractionMetadata(
            numberOfPatternsPerArray=int(numberOfPatternsPerArray),
            numberOfPatternsTotal=int(numberOfPatternsTotal),
            patternDataType=numpy.dtype(patternDataType),
        )
        self._ptychodus_streming_context = self._ptychodus.createStreamingContext(metadata)
        self._ptychodus_streaming_context.start()  # FIXME

    def process(self, pvObject: pvaccess.PvObject) -> pvaccess.PvObject:
        """Processes monitor update"""
        processingBeginTime = time.time()

        (frameId, image, nx, ny, nz, colorMode, fieldKey) = self.reshapeNtNdArray(pvObject)
        frameTimeStamp = TimeUtility.getTimeStampAsFloat(pvObject['timeStamp'])

        if nx is None:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
        else:
            self.logger.debug(f'Frame id {frameId} time stamp {frameTimeStamp}')
            image3d = image[numpy.newaxis, :, :].copy()
            array = ptychodus.api.patterns.SimpleDiffractionPatternArray(
                label=f'Frame{frameId}',
                indexes=numpy.array([frameId]),
                data=image3d,
            )
            self._ptychodus_streaming_context.append_array(array)

        posXQueue = self.metadataQueueMap[self._posXPV]

        while True:
            try:
                posX = posXQueue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                self._ptychodus_streaming_context.append_positions_x(
                    posX['values'],
                    [TimeUtility.getTimeStampAsFloat(ts) for ts in posX['t']],
                )

        posYQueue = self.metadataQueueMap[self._posYPV]

        while True:
            try:
                posY = posYQueue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                self._ptychodus_streaming_context.append_positions_y(
                    posY['values'],
                    [TimeUtility.getTimeStampAsFloat(ts) for ts in posY['t']],
                )

        processingEndTime = time.time()
        self.processingTime += processingEndTime - processingBeginTime
        self.nFramesProcessed += 1

        return pvObject

    def resetStats(self) -> None:
        """Resets statistics for user processor"""
        self.nFramesProcessed = 0
        self.processingTime = 0.0

    def getStats(self) -> dict[str, Any]:
        """Retrieves statistics for user processor"""
        nFramesQueued = self._ptychodus_streaming_context.get_queue_size()
        processedFrameRate = 0.0

        if self.processingTime > 0.0:
            processedFrameRate = self.nFramesProcessed / self.processingTime

        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFramesQueued': nFramesQueued,
            'processingTime': FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate': FloatWithUnits(processedFrameRate, 'fps'),
        }

    def getStatsPvaTypes(self) -> dict[str, pvaccess.ScalarType]:
        """Defines PVA types for different stats variables"""
        return {
            'nFramesProcessed': pvaccess.UINT,
            'nFramesQueued': pvaccess.UINT,
            'processingTime': pvaccess.DOUBLE,
            'processingFrameRate': pvaccess.DOUBLE,
        }
