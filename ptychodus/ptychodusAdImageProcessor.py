from pathlib import Path
from threading import Thread
from typing import Any
import time

import numpy

from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
from pvapy.utility.timeUtility import TimeUtility
import pvaccess as pva

import ptychodus


class PtychodusAdImageProcessor(AdImageProcessor):

    def __init__(self, configDict: dict[str, Any] = {}) -> None:
        super().__init__(configDict)

        settingsFilePath = configDict.get('settingsFilePath')

        self.logger.debug(f'{ptychodus.__name__.title()} ({ptychodus.__version__})')
        from ptychodus.model import ModelArgs, ModelCore

        modelArgs = ModelArgs(
            settingsFilePath=Path(settingsFilePath) if settingsFilePath else None,
            replacementPathPrefix=configDict.get('replacementPathPrefix'),
            rpcPort=-1,
            autoExecuteRPCs=True,
            isDeveloperModeEnabled=False,
        )

        self._ptychodus = ModelCore(modelArgs)
        self._reconstructor = Thread(target=self._ptychodus.batchModeReconstruct)
        self._reconstructFrameId = int(configDict['reconstructFrameId']),
        self._nFramesProcessed = 0
        self._processingTime = 0.
        self.logger.debug(f'Created {type(self).__name__}')

    def start(self):
        '''Called at startup'''
        self._ptychodus.__enter__()

    def stop(self):
        '''Called at shutdown'''
        self._ptychodus.__exit__(None, None, None)

    def configure(self, configDict: dict[str, Any]) -> None:
        '''Configures user processor'''
        self.logger.debug(f'Configuration update: {configDict}')

        numberOfPatternsTotal = configDict['nPatternsTotal']
        numberOfPatternsPerArray = configDict.get('nPatternsPerArray', 1)
        patternDataType = configDict.get('PatternDataType', 'uint16')

        metadata = ptychodus.api.data.DiffractionMetadata(
            numberOfPatternsPerArray=int(numberOfPatternsPerArray),
            numberOfPatternsTotal=int(numberOfPatternsTotal),
            patternDataType=numpy.dtype(patternDataType),
        )
        self._ptychodus.batchModeSetupForStreamingWorkflow(metadata)

    def process(self, pvObject: pva.PvObject) -> pva.PvObject:
        '''Processes monitor update'''
        processingBeginTime = time.time()

        (frameId, image, nx, ny, nz, colorMode, fieldKey) = self.reshapeNtNdArray(pvObject)

        if nx is None:
            self.logger.debug(f'Frame id {frameId} contains an empty image.')
        else:
            array = ptychodus.api.data.SimpleDiffractionPatternArray(
                label=f'Frame{frameId}',
                index=frameId,
                data=image,
                state=ptychodus.api.data.DiffractionPatternState.LOADED,
            )
            self._ptychodus.assembleDiffractionPattern(array)

        processingEndTime = time.time()
        self.processingTime += (processingEndTime - processingBeginTime)
        self.nFramesProcessed += 1

        if frameId > self._reconstructFrameId:
            if not self._reconstructor.is_alive():
                self._reconstructor.start()

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

    def getStatsPvaTypes(self) -> dict[str, pva.ScalarType]:
        '''Defines PVA types for different stats variables'''
        return {
            'nFramesProcessed': pva.UINT,
            'nFramesQueued': pva.UINT,
            'processingTime': pva.DOUBLE,
            'processingFrameRate': pva.DOUBLE,
        }
