from pathlib import Path
import logging

import lightning
import ptychonn

from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNModelProvider:
    def __init__(
        self,
        model_settings: PtychoNNModelSettings,
        training_settings: PtychoNNTrainingSettings,
        *,
        enable_amplitude: bool,
    ) -> None:
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._enable_amplitude = enable_amplitude
        self._model: ptychonn.LitReconSmallModel | None = None
        self._trainer: lightning.Trainer | None = None

    def get_model_name(self) -> str:
        return 'AmplitudePhase' if self._enable_amplitude else 'PhaseOnly'

    def get_num_channels(self) -> int:
        return 2 if self._enable_amplitude else 1

    def get_model(self) -> ptychonn.LitReconSmallModel:
        if (
            self._model is None
            and self._trainer is not None
            and self._trainer.lightning_module is not None
        ):
            return self._trainer.lightning_module
        else:
            logger.debug('Initializing model from settings')
            self._model = ptychonn.LitReconSmallModel(
                nconv=self._model_settings.num_convolution_kernels.get_value(),
                use_batch_norm=self._model_settings.use_batch_normalization.get_value(),
                enable_amplitude=self._enable_amplitude,
                max_lr=float(self._training_settings.max_learning_rate.get_value()),
                min_lr=float(self._training_settings.min_learning_rate.get_value()),
            )

        return self._model

    def open_model(self, file_path: Path) -> None:
        logger.debug(f'Reading model from "{file_path}"')
        self._model = ptychonn.LitReconSmallModel.load_from_checkpoint(file_path)
        self._trainer = None

    def set_trainer(self, trainer: lightning.Trainer) -> None:
        self._model = None
        self._trainer = trainer

    def save_model(self, file_path: Path) -> None:
        if self._trainer is None:
            logger.warning('Need trainer to save model!')
        else:
            logger.debug(f'Writing model to "{file_path}"')
            self._trainer.save_checkpoint(file_path)
