from collections.abc import Iterator, Sequence
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
        reconstructor_chooser: PluginChooser[Reconstructor],
        log_handler: ReconstructorLogHandler,
        reconstructor_api: ReconstructorAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._reconstructor_chooser = reconstructor_chooser
        self._log_handler = log_handler
        self._reconstructor_api = reconstructor_api

        reconstructor_chooser.synchronize_with_parameter(settings.algorithm)
        reconstructor_chooser.add_observer(self)

    def reconstructors(self) -> Iterator[str]:
        for plugin in self._reconstructor_chooser:
            yield plugin.display_name

    def get_reconstructor(self) -> str:
        return self._reconstructor_chooser.get_current_plugin().display_name

    def set_reconstructor(self, name: str) -> None:
        self._reconstructor_chooser.set_current_plugin(name)

    def reconstruct(self, input_product_index: int) -> int:
        return self._reconstructor_api.reconstruct(input_product_index)

    def reconstruct_split(self, input_product_index: int) -> tuple[int, int]:
        return self._reconstructor_api.reconstruct_split(input_product_index)

    def reconstruct_transformed(self, input_product_index: int) -> Sequence[int]:
        return self._reconstructor_api.reconstruct_transformed(input_product_index)

    @property
    def is_reconstructing(self) -> bool:
        return self._reconstructor_api.is_reconstructing

    def flush_log(self) -> Iterator[str]:
        for text in self._log_handler.messages():
            yield text

    def process_results(self, *, block: bool) -> None:
        self._reconstructor_api.process_results(block=block)

    @property
    def is_trainable(self) -> bool:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def get_model_file_filter(self) -> str:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_model_file_filter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def open_model(self, file_path: Path) -> None:
        return self._reconstructor_api.open_model(file_path)

    def save_model(self, file_path: Path) -> None:
        return self._reconstructor_api.save_model(file_path)

    def get_training_data_file_filter(self) -> str:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_training_data_file_filter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def export_training_data(self, file_path: Path, input_product_index: int) -> None:
        return self._reconstructor_api.export_training_data(file_path, input_product_index)

    def get_training_data_path(self) -> Path:
        reconstructor = self._reconstructor_chooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.get_training_data_path()
        else:
            logger.warning('Reconstructor is not trainable!')

        return Path()

    def train(self, data_path: Path) -> TrainOutput:
        return self._reconstructor_api.train(data_path)

    def _update(self, observable: Observable) -> None:
        if observable is self._reconstructor_chooser:
            self.notify_observers()
