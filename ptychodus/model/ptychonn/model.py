from pathlib import Path
import logging

import lightning
import ptychonn

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNModelProvider:

    def __init__(self, modelSettings: PtychoNNModelSettings,
                 trainingSettings: PtychoNNTrainingSettings, *, enableAmplitude: bool) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._enableAmplitude = enableAmplitude
        self._model: ptychonn.LitReconSmallModel | None = None
        self._trainer: lightning.Trainer | None = None

    def getModelName(self) -> str:
        return 'AmplitudePhase' if self._enableAmplitude else 'PhaseOnly'

    def getNumberOfChannels(self) -> int:
        return 2 if self._enableAmplitude else 1

    def getModel(self) -> ptychonn.LitReconSmallModel:
        if self._model is None and self._trainer is not None and self._trainer.lightning_module is not None:
            return self._trainer.lightning_module
        else:
            logger.debug('Initializing model from settings')
            self._model = ptychonn.LitReconSmallModel(
                nconv=self._modelSettings.numberOfConvolutionKernels.value,
                use_batch_norm=self._modelSettings.useBatchNormalization.value,
                enable_amplitude=self._enableAmplitude,
                max_lr=float(self._trainingSettings.maximumLearningRate.value),
                min_lr=float(self._trainingSettings.minimumLearningRate.value),
            )

        return self._model

    def openModel(self, filePath: Path) -> None:
        logger.debug(f'Reading model from \"{filePath}\"')
        self._model = ptychonn.LitReconSmallModel.load_from_checkpoint(filePath)
        self._trainer = None

    def setTrainer(self, trainer: lightning.Trainer) -> None:
        self._model = None
        self._trainer = trainer

    def saveModel(self, filePath: Path) -> None:
        if self._trainer is None:
            logger.warning('Need trainer to save model!')
        else:
            logger.debug(f'Writing model to \"{filePath}\"')
            self._trainer.save_checkpoint(filePath)
