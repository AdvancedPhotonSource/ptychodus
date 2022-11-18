from pathlib import Path
import threading
from typing import Any
import time

import numpy

from pvapy.hpc.adImageProcessor import AdImageProcessor
from pvapy.utility.floatWithUnits import FloatWithUnits
import pvaccess
import pvapy

import ptychodus
import ptychodus.model


class ReconstructionThread(threading.Thread):

    def __init__(self, ptychodus: ptychodus.model.ModelCore, channel: pvapy.Channel) -> None:
        super().__init__()
        self._ptychodus = ptychodus
        self._channel = channel
        self._reconstructEvent = threading.Event()
        self._stopEvent = threading.Event()

    def run(self) -> None:
        while not self._stopEvent.is_set():
            if self._reconstructEvent.wait(timeout=1.):
                self._ptychodus.batchModeReconstruct()
                self._channel.put(0)
                self._reconstructEvent.clear()

    def monitorReconstructPV(self, pv) -> None:  # FIXME typing
        print(f'ReconstructPV :: {type(pv)} :: {pv}')

        if pv['value'] == 1:
            self._reconstructEvent.set()

    def stop(self) -> None:
        self._stopEvent.set()


class PtychodusAdImageProcessor(AdImageProcessor):

    def __init__(self, configDict: dict[str, Any] = {}) -> None:
        super().__init__(configDict)

        settingsFilePath = configDict['settingsFilePath']

        self.logger.debug(f'{ptychodus.__name__.title()} ({ptychodus.__version__})')

        modelArgs = ptychodus.model.ModelArgs(
            restartFilePath=None,
            settingsFilePath=Path(settingsFilePath) if settingsFilePath else None,
            replacementPathPrefix=configDict.get('replacementPathPrefix'),
        )

        reconstructPV = configDict.get('reconstructPV', 'bdpgp:gp:bit3')
        self.logger.debug(f'ReconstructPV: \"{reconstructPV}\"')
        reconstructionChannel = pvapy.Channel(reconstructPV, pvapy.CA)

        self._ptychodus = ptychodus.model.ModelCore(modelArgs)
        self._reconstructionThread = ReconstructionThread(self._ptychodus, reconstructionChannel)
        self._scanXPV = configDict['scanXPV']
        self._scanYPV = configDict['scanYPV']
        self._nFramesProcessed = 0
        self._processingTime = 0.

        reconstructionChannel.monitor(self._reconstructionThread.monitorReconstructPV,
                                      'field(value,alarm,timeStamp)')
        self.logger.debug(f'Created {type(self).__name__}')

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
        self.logger.debug(f'Configuration update: {configDict}')

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

        # The bluesky code is ready to try. It will publish image frames with
        # position attributes (each single frame with pos_x and pos_y attributes)
        # to PVA PV bluesky:image. When bluesky is ready to trigger
        # reconstruction, it will set bdpgp:gp:bit3 to 1. When reconstruction is
        # done, reconstruction should set this back to 0 indicating results are
        # ready. When reconstruction starts, it should make sure this bit is set
        # to 0. The bluesky process will do the same, to ensure there is only one
        # trigger, at the end of image streaming.

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

        index = 0  # FIXME read indexes with positions

        for metadataChannel, metadataQueue in self.metadataQueueMap.items():
            while True:
                try:
                    value = metadataQueue.get(0)
                except pvaccess.QueueEmpty as ex:
                    break
                else:
                    if metadataChannel == self._scanXPV:
                        # TODO rescale value to meters
                        self._ptychodus.assembleScanPositionX(index, value)
                    elif metadataChannel == self._scanYPV:
                        # TODO rescale value to meters
                        self._ptychodus.assembleScanPositionY(index, value)

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
