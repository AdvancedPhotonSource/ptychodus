from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptychonn import ReconSmallModel, Tester, Trainer

from ptychodus.api.geometry import ImageExtent, Point2D
from ptychodus.api.object import ObjectArrayType
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (ReconstructInput, ReconstructOutput,
                                         TrainableReconstructor, TrainOutput)
from ptychodus.api.typing import Float32ArrayType

from ..analysis import ObjectLinearInterpolator, ObjectStitcher
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings, PtychoNNPositionPredictionSettings
 
logger = logging.getLogger(__name__)


class PatternCircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (maxSize, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> PatternCircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: Float32ArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]


class ObjectPatchCircularBuffer:

    def __init__(self, extent: ImageExtent, channels: int, maxSize: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (maxSize, channels, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> ObjectPatchCircularBuffer:
        return cls(ImageExtent(0, 0), 0, 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: ObjectArrayType) -> None:
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]


class PtychoNNTrainableReconstructor(TrainableReconstructor):

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings, 
                 positionPredictionSettings: PtychoNNPositionPredictionSettings, *, enableAmplitude: bool) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._positionPredictionSettings = positionPredictionSettings
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self._enableAmplitude = enableAmplitude
        self._trainingDataFileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        self._modelFileFilterList: list[str] = list()  # TODO

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'AmplitudePhase' if self._enableAmplitude else 'PhaseOnly'

    def _createModel(self) -> ReconSmallModel:
        logger.debug('Building model...')
        return ReconSmallModel(
            nconv=self._modelSettings.numberOfConvolutionKernels.value,
            use_batch_norm=self._modelSettings.useBatchNormalization.value,
            enable_amplitude=self._enableAmplitude,
        )

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        # TODO data size/shape requirements to GUI
        data = parameters.patterns
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) == 0 and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        # TODO extract binning to data loading (and verify that x-y coordinates are correct)
        inputSize = dataSize
        binSize = dataSize // inputSize

        if binSize == 1:
            binnedData = data
        else:
            binnedData = numpy.zeros((data.shape[0], inputSize, inputSize), dtype=data.dtype)

            for i in range(inputSize):
                for j in range(inputSize):
                    binnedData[:, i, j] = numpy.sum(data[:, binSize * i:binSize * (i + 1),
                                                         binSize * j:binSize * (j + 1)])

        logger.debug('Loading model state...')
        tester = Tester(
            model=self._createModel(),
            model_params_path=self._modelSettings.stateFilePath.value,
        )

        logger.debug('Inferring...')
        tester.setTestData(binnedData.astype(numpy.float32),
                           batch_size=self._modelSettings.batchSize.value)
        npzSavePath = None  # TODO self._trainingSettings.outputPath.value / 'preds.npz'
        objectPatches = tester.predictTestData(npz_save_path=npzSavePath)

        logger.debug('Stitching...')
        stitcher = ObjectStitcher(parameters.product.object_.getGeometry())

        for scanPoint, objectPatchChannels in zip(parameters.product.scan, objectPatches):
            patchCenter = Point2D(
                x=scanPoint.positionXInMeters,
                y=scanPoint.positionYInMeters,
            )
            patchArray = numpy.exp(1j * objectPatchChannels[0])

            if objectPatchChannels.shape[0] == 2:
                patchArray *= objectPatchChannels[1]
            else:
                patchArray *= 0.5

            stitcher.addPatch(patchCenter, patchArray)

        product = Product(
            metadata=parameters.product.metadata,
            scan=parameters.product.scan,
            probe=parameters.product.probe,
            object_=stitcher.build(),
            costs=list(),  # TODO put something here?
        )

        return ReconstructOutput(product, 0)

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        interpolator = ObjectLinearInterpolator(parameters.product.object_)
        probeExtent = parameters.product.probe.getExtent()

        if self._patternBuffer.isZeroSized:
            patternExtent = ImageExtent(widthInPixels=parameters.patterns.shape[-1],
                                        heightInPixels=parameters.patterns.shape[-2])
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.value)
            channels = 2 if self._enableAmplitude else 1
            self._patternBuffer = PatternCircularBuffer(patternExtent, maximumSize)
            self._objectPatchBuffer = ObjectPatchCircularBuffer(patternExtent, channels,
                                                                maximumSize)

        for scanPoint in parameters.product.scan:
            patchCenter = Point2D(
                x=scanPoint.positionXInMeters,
                y=scanPoint.positionYInMeters,
            )
            objectPatch = interpolator.getPatch(patchCenter, probeExtent)
            self._objectPatchBuffer.append(objectPatch.array)

        for pattern in parameters.patterns.astype(numpy.float32):
            self._patternBuffer.append(pattern)

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._trainingDataFileFilterList

    def getSaveTrainingDataFileFilter(self) -> str:
        return self._trainingDataFileFilterList[0]

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            'diffractionPatterns': self._patternBuffer.getBuffer(),
            'objectPatches': self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

    def train(self) -> TrainOutput:
        outputPath = self._trainingSettings.outputPath.value \
                if self._trainingSettings.saveTrainingArtifacts.value else None

        trainer = Trainer(
            model=self._createModel(),
            batch_size=self._modelSettings.batchSize.value,
            output_path=outputPath,
            output_suffix=self._trainingSettings.outputSuffix.value,
        )

        trainer.setTrainingData(
            X_train_full=self._patternBuffer.getBuffer(),
            Y_ph_train_full=self._objectPatchBuffer.getBuffer(),
            valid_data_ratio=float(self._trainingSettings.validationSetFractionalSize.value),
        )
        trainer.setOptimizationParams(
            epochs_per_half_cycle=self._trainingSettings.optimizationEpochsPerHalfCycle.value,
            max_lr=float(self._trainingSettings.maximumLearningRate.value),
            min_lr=float(self._trainingSettings.minimumLearningRate.value),
        )

        logger.debug('Loading model state...')
        trainer.initModel()

        logger.debug('Training...')
        trainer.run(
            epochs=self._trainingSettings.trainingEpochs.value,
            output_frequency=self._trainingSettings.statusIntervalInEpochs.value,
        )

        return TrainOutput(
            trainingLoss=[losses[0] for losses in trainer.metrics['losses']],
            validationLoss=[losses[0] for losses in trainer.metrics['val_losses']],
            result=0,
        )

    def clearTrainingData(self) -> None:
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return self._modelFileFilterList

    def getSaveModelFileFilter(self) -> str:
        return self._modelFileFilterList[0]

    def saveModel(self, filePath: Path) -> None:
        raise NotImplementedError(f'Save trained model to \"{filePath}\"')  # TODO
    
    
