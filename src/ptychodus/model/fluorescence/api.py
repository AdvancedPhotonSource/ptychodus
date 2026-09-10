from collections.abc import Iterator
from pathlib import Path
import logging

from ptychodus.api.fluorescence import (
    FluorescenceEnhancer,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)
from ptychodus.api.plugins import PluginChooser

from ..product import ProductAPI
from ..task_manager import TaskManager
from .monitor import EnhanceFluorescenceBackgroundTask, FluorescenceTaskMonitor
from .repository import (
    FluorescenceItemState,
    FluorescenceRepository,
    FluorescenceRepositoryItem,
)
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


class FluorescenceAPI:
    def __init__(
        self,
        task_manager: TaskManager,
        settings: FluorescenceSettings,
        product_api: ProductAPI,
        repository: FluorescenceRepository,
        enhancer_chooser: PluginChooser[FluorescenceEnhancer],
        task_monitor: FluorescenceTaskMonitor,
        file_reader_chooser: PluginChooser[FluorescenceFileReader],
        file_writer_chooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self._task_manager = task_manager
        self._settings = settings
        self._product_api = product_api
        self._repository = repository
        self._enhancer_chooser = enhancer_chooser
        self._task_monitor = task_monitor
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser

    def get_task_monitor(self) -> FluorescenceTaskMonitor:
        return self._task_monitor

    def get_item(self, item_index: int) -> FluorescenceRepositoryItem:
        return self._repository[item_index]

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

    def open_measured_dataset(
        self, file_path: Path, product_index: int, *, file_type: str | None = None
    ) -> int:
        if not file_path.is_file():
            raise FileNotFoundError(
                f'Refusing to load dataset from invalid file path "{file_path}"'
            )

        # Resolve the product first — a bad product_index should fail before we
        # touch the file, so an orphaned item is never inserted.
        product_item = self._product_api.get_item(product_index)

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

        name = self._repository.create_unique_name(file_path.stem)
        item = FluorescenceRepositoryItem(
            self._repository,
            name=name,
            product=product_item,
            measured=dataset,
            source_path=file_path,
            source_file_type=resolved_type,
        )
        return self._repository.insert_item(item)

    def remove_item(self, item_index: int) -> None:
        self._repository.remove_item(item_index)

    def sync_to_settings(self, item_index: int) -> None:
        """Point the fluorescence settings at this item's source file."""
        try:
            item = self._repository[item_index]
        except IndexError:
            logger.warning(f'Failed to look up fluorescence item {item_index}!')
            return

        source_path = item.get_source_path()

        if source_path is None:
            logger.warning(f'Item "{item.get_name()}" has no source path to sync!')
            return

        self._settings.file_path.set_value(source_path)
        source_file_type = item.get_source_file_type()

        if source_file_type is not None:
            self._settings.file_type.set_value(source_file_type)

    def save_enhanced_dataset(
        self, item_index: int, file_path: Path, *, file_type: str | None = None
    ) -> None:
        try:
            item = self._repository[item_index]
        except IndexError as exc:
            raise ValueError(f'No fluorescence item at index {item_index}') from exc

        dataset = item.get_enhanced()
        if dataset is None:
            raise ValueError(f'Item "{item.get_name()}" has no enhanced dataset to save')

        if file_type is not None:
            self._file_writer_chooser.set_current_plugin(file_type)

        resolved_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{resolved_type}"')
        writer = self._file_writer_chooser.get_current_plugin().strategy
        writer.write(file_path, dataset)

    def enhance(
        self,
        item_index: int,
        *,
        algorithm: str | None = None,
        output_file_path: Path | None = None,
        output_file_type: str | None = None,
        block: bool = False,
    ) -> None:
        try:
            item = self._repository[item_index]
        except IndexError as exc:
            raise ValueError(f'No fluorescence item at index {item_index}') from exc

        if item.get_state() is FluorescenceItemState.ORPHANED:
            raise RuntimeError(
                f'Cannot enhance item "{item.get_name()}": target product was removed'
            )

        if self._task_monitor.is_processing:
            raise RuntimeError('Fluorescence enhancement is already in progress')

        if algorithm is not None:
            self._enhancer_chooser.set_current_plugin(algorithm)

        output_file_writer: FluorescenceFileWriter | None = None

        if output_file_path is not None:
            if output_file_type is not None:
                self._file_writer_chooser.set_current_plugin(output_file_type)

            output_file_writer = self._file_writer_chooser.get_current_plugin().strategy

        product = item.get_product().get_product()
        item.set_state(FluorescenceItemState.ENHANCING)

        background_task = EnhanceFluorescenceBackgroundTask(
            self._task_monitor,
            self._enhancer_chooser.get_current_plugin().strategy,
            item.get_measured(),
            product,
            item,
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
