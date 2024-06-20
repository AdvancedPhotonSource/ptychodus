from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
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
from .common import MODEL_FILE_FILTER, TRAINING_DATA_FILE_FILTER
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNTrainableReconstructor(TrainableReconstructor):

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings, *, enableAmplitude: bool) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self._enableAmplitude = enableAmplitude

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'AmplitudePhase' if self._enableAmplitude else 'PhaseOnly'

    def _createModel(self) -> ptychonn.LitReconSmallModel:
        # TODO keep model in memory
        logger.debug('Building model...')

        if self._modelSettings.modelFilePath.value.exists():
            path = self._modelSettings.modelFilePath.value
            parameters = None
        else:
            path = None
            parameters = dict(
                nconv=self._modelSettings.numberOfConvolutionKernels.value,
                use_batch_norm=self._modelSettings.useBatchNormalization.value,
                enable_amplitude=self._enableAmplitude,
                max_lr=float(self._trainingSettings.maximumLearningRate.value),
                min_lr=float(self._trainingSettings.minimumLearningRate.value),
            )

        return ptychonn.init_or_load_model(
            model_type=ptychonn.LitReconSmallModel,
            model_checkpoint_path=path,
            model_init_params=parameters,
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
        model = self._createModel()

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
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.value)
            channels = 2 if self._enableAmplitude else 1
            self._patternBuffer = PatternCircularBuffer(patternExtent, maximumSize)
            self._objectPatchBuffer = ObjectPatchCircularBuffer(patternExtent, channels,
                                                                maximumSize)

        for scanPoint in parameters.product.scan:
            objectPatch = interpolator.getPatch(scanPoint, probeExtent)
            self._objectPatchBuffer.append(objectPatch.array)

        for pattern in parameters.patterns.astype(numpy.float32):
            self._patternBuffer.append(pattern)

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return [self.getSaveTrainingDataFileFilter()]

    def getSaveTrainingDataFileFilter(self) -> str:
        return TRAINING_DATA_FILE_FILTER

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            'diffractionPatterns': self._patternBuffer.getBuffer(),
            'objectPatches': self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

    def train(self) -> TrainOutput:
        logger.debug("Loading model state...")
        model = self._createModel()

        logger.debug("Training...")
        trainingSetFractionalSize = 1 - self._trainingSettings.validationSetFractionalSize.value
        trainer, trainerLog = ptychonn.train(
            model=model,
            batch_size=self._modelSettings.batchSize.value,
            out_dir=None,
            X_train=self._patternBuffer.getBuffer(),
            Y_train=self._objectPatchBuffer.getBuffer(),
            epochs=self._trainingSettings.trainingEpochs.value,
            training_fraction=float(trainingSetFractionalSize),
            log_frequency=self._trainingSettings.statusIntervalInEpochs.value,
        )

        if self._trainingSettings.saveTrainingArtifacts.value:
            ptychonn.create_model_checkpoint(
                trainer,
                self._trainingSettings.trainingArtifactsDirectory.value /
                self._trainingSettings.trainingArtifactsSuffix.value,
            )

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

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return [self.getSaveModelFileFilter()]

    def getSaveModelFileFilter(self) -> str:
        return MODEL_FILE_FILTER

    def saveModel(self, filePath: Path) -> None:
        raise NotImplementedError(f'Save trained model to \"{filePath}\"')  # TODO
