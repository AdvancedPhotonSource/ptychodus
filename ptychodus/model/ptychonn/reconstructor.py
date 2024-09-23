from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Final
import logging

import numpy
import numpy.typing
import ptychonn

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (ReconstructInput, ReconstructOutput,
                                         TrainableReconstructor, TrainOutput)

from ..analysis import ObjectLinearInterpolator, ObjectStitcher
from .buffers import ObjectPatchCircularBuffer, PatternCircularBuffer
from .model import PtychoNNModelProvider
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNTrainableReconstructor(TrainableReconstructor):
    MODEL_FILE_FILTER: Final[str] = 'PyTorch Lightning Checkpoint Files (*.ckpt)'
    TRAINING_DATA_FILE_FILTER: Final[str] = 'NumPy Zipped Archive (*.npz)'
    PATCHES_KW: Final[str] = 'real'
    PATTERNS_KW: Final[str] = 'reciprocal'

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings,
                 modelProvider: PtychoNNModelProvider) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._modelProvider = modelProvider
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return self._modelProvider.getModelName()

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

        model = self._modelProvider.getModel()

        logger.debug('Inferring...')
        objectPatches = ptychonn.infer(
            data=binnedData.astype(numpy.float32),
            model=model,
        )

        logger.debug('Stitching...')
        stitcher = ObjectStitcher(parameters.product.object_.getGeometry())

        for scanPoint, objectPatchChannels in zip(parameters.product.scan, objectPatches):
            patchArray = numpy.exp(1j * objectPatchChannels[0])

            if objectPatchChannels.shape[0] == 2:
                patchArray *= objectPatchChannels[1]
            else:
                patchArray *= 0.5

            stitcher.addPatch(scanPoint, patchArray)

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
            patternExtent = ImageExtent(
                widthInPixels=parameters.patterns.shape[-1],
                heightInPixels=parameters.patterns.shape[-2],
            )
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.getValue())
            self._patternBuffer = PatternCircularBuffer(patternExtent, maximumSize)
            self._objectPatchBuffer = ObjectPatchCircularBuffer(
                patternExtent, self._modelProvider.getNumberOfChannels(), maximumSize)

        for scanPoint in parameters.product.scan:
            objectPatch = interpolator.getPatch(scanPoint, probeExtent)
            self._objectPatchBuffer.append(objectPatch.array)

        for pattern in parameters.patterns.astype(numpy.float32):
            self._patternBuffer.append(pattern)

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        return [self.getOpenTrainingDataFileFilter()]

    def getOpenTrainingDataFileFilter(self) -> str:
        return self.TRAINING_DATA_FILE_FILTER

    def openTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Reading \"{filePath}\" as \"NPZ\"')
        trainingData = numpy.load(filePath)
        self._patternBuffer.setBuffer(trainingData[self.PATTERNS_KW])
        self._objectPatchBuffer.setBuffer(trainingData[self.PATCHES_KW])

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return [self.getSaveTrainingDataFileFilter()]

    def getSaveTrainingDataFileFilter(self) -> str:
        return self.TRAINING_DATA_FILE_FILTER

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            self.PATTERNS_KW: self._patternBuffer.getBuffer(),
            self.PATCHES_KW: self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez_compressed(filePath, **trainingData)

    def train(self) -> TrainOutput:
        model = self._modelProvider.getModel()
        logger.debug('Training...')
        trainingSetFractionalSize = 1 - self._trainingSettings.validationSetFractionalSize.getValue(
        )
        trainer, trainerLog = ptychonn.train(
            model=model,
            batch_size=self._modelSettings.batchSize.getValue(),
            out_dir=None,
            X_train=self._patternBuffer.getBuffer(),
            Y_train=self._objectPatchBuffer.getBuffer(),
            epochs=self._trainingSettings.trainingEpochs.getValue(),
            training_fraction=float(trainingSetFractionalSize),
            log_frequency=self._trainingSettings.statusIntervalInEpochs.getValue(),
            strategy='ddp_notebook',
        )
        self._modelProvider.setTrainer(trainer)

        trainingLoss: list[float] = list()
        validationLoss: list[float] = list()

        for entry in trainerLog.logs:
            try:
                tloss = entry['training_loss']
                vloss = entry['validation_loss']
            except KeyError:
                pass
            else:
                trainingLoss.append(tloss)
                validationLoss.append(vloss)

        return TrainOutput(
            trainingLoss=trainingLoss,
            validationLoss=validationLoss,
            result=0,
        )

    def clearTrainingData(self) -> None:
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        return [self.getOpenModelFileFilter()]

    def getOpenModelFileFilter(self) -> str:
        return self.MODEL_FILE_FILTER

    def openModel(self, filePath: Path) -> None:
        self._modelProvider.openModel(filePath)

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return [self.getSaveModelFileFilter()]

    def getSaveModelFileFilter(self) -> str:
        return self.MODEL_FILE_FILTER

    def saveModel(self, filePath: Path) -> None:
        self._modelProvider.saveModel(filePath)
