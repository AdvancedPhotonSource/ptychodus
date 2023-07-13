from __future__ import annotations
import logging

from ptychonn import Trainer
import numpy
import numpy.typing

from ...api.data import DiffractionPatternArrayType
from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
from .factory import PtychoNNModelFactory
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNPhaseOnlyTrainer(TrainableReconstructor):

    def __init__(self, settings: PtychoNNModelSettings, trainingSettings: PtychoNNTrainingSettings,
                 factory: PtychoNNModelFactory) -> None:
        super().__init__()
        self._settings = settings
        self._trainingSettings = trainingSettings
        self._factory = factory
        self._diffractionPatternsArray: DiffractionPatternArrayType | None = None
        self._objectPatchesArray: ObjectArrayType | None = None

    @property
    def name(self) -> str:
        return 'TrainPhase'

    def _appendDiffractionPatterns(self, array: DiffractionPatternArrayType) -> None:
        if self._diffractionPatternsArray is None:
            self._diffractionPatternsArray = array
        else:
            self._diffractionPatternsArray = numpy.concatenate(
                (self._diffractionPatternsArray, array))

    def _appendObjectPatches(self, array: ObjectArrayType) -> None:
        if self._objectPatchesArray is None:
            self._objectPatchesArray = array
        else:
            self._objectPatchesArray = numpy.concatenate((self._objectPatchesArray, array))

    def execute(self, parameters: ReconstructInput) -> ReconstructOutput:
        objectPatchesList: list[ObjectArrayType] = list()
        objectInterpolator = parameters.objectInterpolator

        probeExtent = ImageExtent(
            width=parameters.probeArray.shape[-1],
            height=parameters.probeArray.shape[-2],
        )

        for scanPoint in parameters.scan.values():
            objectPatch = objectInterpolator.getPatch(scanPoint, probeExtent)
            objectPatchesList.append(objectPatch.array)

        self._appendDiffractionPatterns(parameters.diffractionPatternArray)
        self._appendObjectPatches(numpy.concatenate(objectPatchesList, axis=0))

        return ReconstructOutput.createNull()

    def train(self) -> None:
        if self._diffractionPatternsArray is None or self._objectPatchesArray is None:
            raise ValueError('Missing training data!')

        outputPath = self._trainingSettings.outputPath.value \
                if self._trainingSettings.saveTrainingArtifacts.value else None

        trainer = Trainer(
            model=self._factory.createModel(),
            batch_size=self._settings.batchSize.value,
            output_path=outputPath,
            output_suffix=self._trainingSettings.outputSuffix.value,
        )
        trainer.setTrainingData(
            X_train_full=self._diffractionPatternsArray.astype(numpy.float32),
            Y_ph_train_full=self._objectPatchesArray.astype(numpy.float32),
            valid_data_ratio=float(self._trainingSettings.validationSetFractionalSize.value),
        )
        trainer.setOptimizationParams(
            epochs_per_half_cycle=self._trainingSettings.optimizationEpochsPerHalfCycle.value,
            max_lr=float(self._trainingSettings.maximumLearningRate.value),
            min_lr=float(self._trainingSettings.minimumLearningRate.value),
        )

        logger.debug('Loading model state...')
        trainer.initModel(model_params_path=self._settings.stateFilePath.value)

        logger.debug('Training...')
        trainer.run(
            epochs=self._trainingSettings.trainingEpochs.value,
            output_frequency=self._trainingSettings.statusIntervalInEpochs.value,
        )

    def clear(self) -> None:
        self._diffractionPatternsArray = None
        self._objectPatchesArray = None
