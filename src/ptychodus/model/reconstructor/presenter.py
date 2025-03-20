from collections.abc import Iterator
from pathlib import Path
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    Reconstructor,
    TrainableReconstructor,
    TrainOutput,
)

from .api import ReconstructorAPI
from .log import ReconstructorLogHandler
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):
    def __init__(
        self,
        settings: ReconstructorSettings,
        reconstructorChooser: PluginChooser[Reconstructor],
        logHandler: ReconstructorLogHandler,
        reconstructorAPI: ReconstructorAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._reconstructorChooser = reconstructorChooser
        self._logHandler = logHandler
        self._reconstructorAPI = reconstructorAPI

        reconstructorChooser.synchronize_with_parameter(settings.algorithm)
        reconstructorChooser.add_observer(self)

    def reconstructors(self) -> Iterator[str]:
        for plugin in self._reconstructorChooser:
            yield plugin.display_name

    def get_reconstructor(self) -> str:
        return self._reconstructorChooser.get_current_plugin().display_name

    def set_reconstructor(self, name: str) -> None:
        self._reconstructorChooser.set_current_plugin(name)

    def reconstruct(self, inputProductIndex: int) -> int:
        return self._reconstructorAPI.reconstruct(inputProductIndex)

    def reconstruct_split(self, inputProductIndex: int) -> tuple[int, int]:
        return self._reconstructorAPI.reconstruct_split(inputProductIndex)

    @property
    def is_reconstructing(self) -> bool:
        return self._reconstructorAPI.is_reconstructing

    def flush_log(self) -> Iterator[str]:
        for text in self._logHandler.messages():
            yield text

    def process_results(self, *, block: bool) -> None:
        self._reconstructorAPI.process_results(block=block)

    @property
    def is_trainable(self) -> bool:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def get_model_file_filter(self) -> str:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_model_file_filter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def open_model(self, filePath: Path) -> None:
        return self._reconstructorAPI.open_model(filePath)

    def save_model(self, filePath: Path) -> None:
        return self._reconstructorAPI.save_model(filePath)

    def get_training_data_file_filter(self) -> str:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_training_data_file_filter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def export_training_data(self, filePath: Path, inputProductIndex: int) -> None:
        return self._reconstructorAPI.export_training_data(filePath, inputProductIndex)

    def get_training_data_path(self) -> Path:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_training_data_path()
        else:
            logger.warning('Reconstructor is not trainable!')

        return Path()

    def train(self, dataPath: Path) -> TrainOutput:
        return self._reconstructorAPI.train(dataPath)

    def _update(self, observable: Observable) -> None:
        if observable is self._reconstructorChooser:
            self.notify_observers()
