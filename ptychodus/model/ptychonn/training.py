from __future__ import annotations
from typing import TypeAlias
import logging

from ptychonn import Trainer
import numpy
import numpy.typing

from ...api.image import ImageExtent
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
from .factory import PtychoNNModelFactory
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

FloatArrayType: TypeAlias = numpy.typing.NDArray[numpy.float32]

logger = logging.getLogger(__name__)


class CircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros((maxSize, *extent.shape), dtype=numpy.float32)
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> CircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: FloatArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]


class PtychoNNPhaseOnlyTrainer(TrainableReconstructor):

    def __init__(self, settings: PtychoNNModelSettings, trainingSettings: PtychoNNTrainingSettings,
                 factory: PtychoNNModelFactory) -> None:
        super().__init__()
        self._settings = settings
        self._trainingSettings = trainingSettings
        self._factory = factory
        self._diffractionPatternBuffer = CircularBuffer.createZeroSized()
        self._objectPhasePatchBuffer = CircularBuffer.createZeroSized()

    @property
    def name(self) -> str:
        return 'TrainPhase'

    def execute(self, parameters: ReconstructInput) -> ReconstructOutput:
        probeExtent = ImageExtent(
            width=parameters.probeArray.shape[-1],
            height=parameters.probeArray.shape[-2],
        )
        objectInterpolator = parameters.objectInterpolator

        if self._diffractionPatternBuffer.isZeroSized:
            diffractionPatternExtent = ImageExtent(
                width=parameters.diffractionPatternArray.shape[-1],
                height=parameters.diffractionPatternArray.shape[-2],
            )
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.value)

            self._diffractionPatternBuffer = CircularBuffer(diffractionPatternExtent, maximumSize)
            self._objectPhasePatchBuffer = CircularBuffer(diffractionPatternExtent, maximumSize)

        for scanIndex, scanPoint in parameters.scan.items():
            objectPatch = objectInterpolator.getPatch(scanPoint, probeExtent)
            objectPhasePatch = numpy.angle(objectPatch.array).astype(numpy.float32)
            # FIXME save phase patch
            self._objectPhasePatchBuffer.append(objectPhasePatch)

        for pattern in parameters.diffractionPatternArray.astype(numpy.float32):
            self._diffractionPatternBuffer.append(pattern)

        return ReconstructOutput.createNull()

    def train(self) -> None:
        outputPath = self._trainingSettings.outputPath.value \
                if self._trainingSettings.saveTrainingArtifacts.value else None

        trainer = Trainer(
            model=self._factory.createModel(),
            batch_size=self._settings.batchSize.value,
            output_path=outputPath,
            output_suffix=self._trainingSettings.outputSuffix.value,
        )

        trainer.setTrainingData(
            X_train_full=self._diffractionPatternBuffer.getBuffer(),
            Y_ph_train_full=self._objectPhasePatchBuffer.getBuffer(),
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
        # TODO ptychonn.plot.plot_metrics(trainer.metrics)

    def reset(self) -> None:
        self._diffractionPatternBuffer = CircularBuffer.createZeroSized()
        self._objectPhasePatchBuffer = CircularBuffer.createZeroSized()
