from pathlib import Path
from typing import Any
import logging
import threading
import time

import numpy

from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
import pvaccess
import pvapy

import ptychodus
import ptychodus.model


class ReconstructionThread(threading.Thread):

    def __init__(self, ptychodus: ptychodus.model.ModelCore, reconstructPV: str) -> None:
        super().__init__()
        self._ptychodus = ptychodus
        self._channel = pvapy.Channel(reconstructPV, pvapy.CA)
        self._reconstructEvent = threading.Event()
        self._stopEvent = threading.Event()

        self._channel.subscribe('reconstructor', self._monitor)
        self._channel.startMonitor()
        logging.debug(
            f'{self._channel.getName()} {self._channel.isConnected()} {self._channel.isMonitorActive()}'
        )

    def run(self) -> None:
        while not self._stopEvent.is_set():
            if self._reconstructEvent.wait(timeout=1.):
                self._ptychodus.batchModeReconstruct()
                self._reconstructEvent.clear()
                # When reconstruction is done, reconstruction should set this back to 0 indicating results are ready.
                self._channel.put(0)

    def _monitor(self, pv) -> None:
        # NOTE caput bdpgp:gp:bit3 1
        logging.debug(f'{type(pv)} :: {pv}')

        # When bluesky is ready to trigger reconstruction, it will set bdpgp:gp:bit3 to 1.
        # When reconstruction starts, it should make sure this bit is set to 0.
        # The bluesky process will do the same, to ensure there is only one trigger, at the end of image streaming.
        if pv['value'] == 1:
            self._reconstructEvent.set()

    def stop(self) -> None:
        self._stopEvent.set()


class PtychodusAdImageProcessor(AdImageProcessor):

    def __init__(self, configDict: dict[str, Any] = {}) -> None:
        super().__init__(configDict)

        settingsFilePath = configDict['settingsFilePath']
        reconstructPV = configDict.get('reconstructPV', 'bdpgp:gp:bit3')

        self.logger.debug(f'{ptychodus.__name__.title()} ({ptychodus.__version__})')

        modelArgs = ptychodus.model.ModelArgs(
            restartFilePath=None,
            settingsFilePath=Path(settingsFilePath) if settingsFilePath else None,
            replacementPathPrefix=configDict.get('replacementPathPrefix'),
        )

        self._ptychodus = ptychodus.model.ModelCore(modelArgs)
        self._reconstructionThread = ReconstructionThread(self._ptychodus, reconstructPV)
        self._posXPV = configDict.get('posXPV', 'bluesky:pos_x')
        self._posYPV = configDict.get('posYPV', 'bluesky:pos_y')
        self._nFramesProcessed = 0
        self._processingTime = 0.

    def start(self):
        '''Called at startup'''
        self._ptychodus.__enter__()
        self._reconstructionThread.start()

    def stop(self):
        '''Called at shutdown'''
        self._reconstructionThread.stop()
        self._reconstructionThread.join()
        self._ptychodus.__exit__(None, None, None)

    def configure(self, configDict: dict[str, Any]) -> None:
        '''Configures user processor'''
        numberOfPatternsTotal = configDict['nPatternsTotal']
        numberOfPatternsPerArray = configDict.get('nPatternsPerArray', 1)
        patternDataType = configDict.get('PatternDataType', 'uint16')

        metadata = ptychodus.api.data.DiffractionMetadata(
            numberOfPatternsPerArray=int(numberOfPatternsPerArray),
            numberOfPatternsTotal=int(numberOfPatternsTotal),
            patternDataType=numpy.dtype(patternDataType),
        )
        self._ptychodus.resetStreamingWorkflow(metadata)

    def process(self, pvObject: pvaccess.PvObject) -> pvaccess.PvObject:
        '''Processes monitor update'''
        processingBeginTime = time.time()

        (frameId, image, nx, ny, nz, colorMode, fieldKey) = self.reshapeNtNdArray(pvObject)

        if nx is None:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
        else:
            image3d = image[numpy.newaxis, :, :].copy()
            array = ptychodus.api.data.SimpleDiffractionPatternArray(
                label=f'Frame{frameId}',
                index=frameId,
                data=image3d,
                state=ptychodus.api.data.DiffractionPatternState.LOADED,
            )
            self._ptychodus.assembleDiffractionPattern(array)

        posXQueue = self.metadataQueueMap[self._posXPV]

        while True:
            try:
                posX = posXQueue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                # TODO rescale value to meters, values is a numpy.array(list[float]), t is list[time_t]
                # FIXME need to match pos[t] with frame numbers
                self._ptychodus.assembleScanPositionsX(posX['values'], posX['t'])

        posYQueue = self.metadataQueueMap[self._posYPV]

        while True:
            try:
                posY = posYQueue.get(0)
            except pvaccess.QueueEmpty:
                break
            else:
                # TODO rescale value to meters, values is a numpy.array(list[float]), t is list[time_t]
                # FIXME need to match pos[t] with frame numbers
                self._ptychodus.assembleScanPositionsY(posY['values'], posY['t'])

        processingEndTime = time.time()
        self.processingTime += (processingEndTime - processingBeginTime)
        self.nFramesProcessed += 1

        return pvObject

    def resetStats(self) -> None:
        '''Resets statistics for user processor'''
        self.nFramesProcessed = 0
        self.processingTime = 0.

    def getStats(self) -> dict[str, Any]:
        '''Retrieves statistics for user processor'''
        nFramesQueued = self._ptychodus.getDiffractionPatternAssemblyQueueSize()
        processedFrameRate = 0.

        if self.processingTime > 0.:
            processedFrameRate = self.nFramesProcessed / self.processingTime

        return {
            'nFramesProcessed': self.nFramesProcessed,
            'nFramesQueued': nFramesQueued,
            'processingTime': FloatWithUnits(self.processingTime, 's'),
            'processedFrameRate': FloatWithUnits(processedFrameRate, 'fps'),
        }

    def getStatsPvaTypes(self) -> dict[str, pvaccess.ScalarType]:
        '''Defines PVA types for different stats variables'''
        return {
            'nFramesProcessed': pvaccess.UINT,
            'nFramesQueued': pvaccess.UINT,
            'processingTime': pvaccess.DOUBLE,
            'processingFrameRate': pvaccess.DOUBLE,
        }
