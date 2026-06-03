from collections.abc import Iterator
from pathlib import Path
import logging

from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.plugins import PluginChooser

from ..product import ProductAPI
from ..task_manager import TaskManager
from .monitor import (
    EnhanceFluorescenceBackgroundTask,
    FluorescenceDatasetEmitter,
    FluorescenceTaskMonitor,
)
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


class FluorescenceAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        settings: FluorescenceSettings,
        product_api: ProductAPI,
        enhancer_chooser: PluginChooser[FluorescenceEnhancer],
        task_monitor: FluorescenceTaskMonitor,
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self._task_manager = task_manager
        self._settings = settings
        self._product_api = product_api
        self._enhancer_chooser = enhancer_chooser
        self._task_monitor = task_monitor
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser

    def get_dataset_emitter(self) -> FluorescenceDatasetEmitter:
        return self._task_monitor.get_dataset_emitter()

    def get_product_name(self, product_index: int) -> str:
        return self._product_api.get_item(product_index).get_name()

    def get_pixel_geometry(self, product_index: int) -> PixelGeometry:
        item = self._product_api.get_item(product_index)
        return item.get_geometry().get_object_plane_pixel_geometry()

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def load_measured_dataset(
        self, file_path: Path, *, file_type: str | None = None
    ) -> FluorescenceDataset:
        if not file_path.is_file():
            raise FileNotFoundError(
                f'Refusing to load dataset from invalid file path "{file_path}"'
            )

        if file_type is not None:
            self._file_reader_chooser.set_current_plugin(file_type)

        resolved_type = self._file_reader_chooser.get_current_plugin().simple_name
        logger.debug(f'Reading "{file_path}" as "{resolved_type}"')
        file_reader = self._file_reader_chooser.get_current_plugin().strategy

        try:
            dataset = file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        self._settings.file_path.set_value(file_path)
        return dataset

    def save_enhanced_dataset(
        self, dataset: FluorescenceDataset, file_path: Path, *, file_type: str | None = None
    ) -> None:
        if file_type is not None:
            self._file_writer_chooser.set_current_plugin(file_type)

        resolved_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{resolved_type}"')
        writer = self._file_writer_chooser.get_current_plugin().strategy
        writer.write(file_path, dataset)

    def enhance(
        self,
        product_index: int,
        dataset: FluorescenceDataset,
        *,
        algorithm: str | None = None,
        output_file_path: Path | None = None,
        output_file_type: str | None = None,
        block: bool = False,
    ) -> None:
        if algorithm is not None:
            self._enhancer_chooser.set_current_plugin(algorithm)

        output_file_writer: FluorescenceFileWriter | None = None

        if output_file_path is not None:
            if output_file_type is not None:
                self._file_writer_chooser.set_current_plugin(output_file_type)

            output_file_writer = self._file_writer_chooser.get_current_plugin().strategy

        product = self._product_api.get_item(product_index).get_product()
        background_task = EnhanceFluorescenceBackgroundTask(
            self._task_monitor,
            self._enhancer_chooser.get_current_plugin().strategy,
            dataset,
            product,
            output_file_path,
            output_file_writer,
        )
        snapshot = self._task_monitor.get_completed_runs()
        self._task_manager.put_background_task(background_task)

        if block:
            while not self._task_manager.is_stopping:
                if self._task_monitor.wait_for_completion_after(
                    snapshot, timeout=TaskManager.WAIT_TIME_S
                ):
                    self._task_manager.run_foreground_tasks()
                    break

    def enhance_local(
        self,
        product_index: int,
        input_file_path: Path,
        output_file_path: Path,
        *,
        input_file_type: str | None = None,
        output_file_type: str | None = None,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        dataset = self.load_measured_dataset(input_file_path, file_type=input_file_type)
        self.enhance(
            product_index,
            dataset,
            algorithm=algorithm,
            output_file_path=output_file_path,
            output_file_type=output_file_type,
            block=block,
        )
