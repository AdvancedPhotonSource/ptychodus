from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.diffraction import Polarization
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter

from ..diffraction import AssembledDiffractionDataset
from ..task_manager import TaskManager
from .item import ProductRepositoryItem, ProductState
from .item_factory import ProductRepositoryItemFactory
from .object.builder_factory import ObjectBuilderFactory
from .object.settings import ObjectSettings
from .object_repository import ObjectRepository
from .probe.builder_factory import ProbeBuilderFactory
from .probe.settings import ProbeSettings
from .probe_repository import ProbeRepository
from .repository import ProductRepository
from .probe_positions.builder_factory import ProbePositionsBuilderFactory
from .probe_positions.settings import ProbePositionsSettings
from .scan_repository import ProbePositionsRepository
from .settings import ProductSettings

logger = logging.getLogger(__name__)


class PositionsStreamingContext:
    def __init__(self) -> None:
        self._positions_x_m: list[float] = []
        self._triggers_x: list[int] = []
        self._positions_y_m: list[float] = []
        self._triggers_y: list[int] = []

    def start(self) -> None:
        self._positions_x_m.clear()
        self._triggers_x.clear()
        self._positions_y_m.clear()
        self._triggers_y.clear()

    def append_positions_x(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_x_m.extend(values_m)
        self._triggers_x.extend(trigger_counts)

    def append_positions_y(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_y_m.extend(values_m)
        self._triggers_y.extend(trigger_counts)

    def stop(self) -> None:
        pass  # TODO


class ProbePositionsAPI:
    def __init__(
        self,
        settings: ProbePositionsSettings,
        repository: ProbePositionsRepository,
        builder_factory: ProbePositionsBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory

    def create_streaming_context(self) -> PositionsStreamingContext:
        return PositionsStreamingContext()

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_probe_positions(
        self, index: int, builder_name: str, builder_parameters: Mapping[str, Any] | None = None
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builder_factory.create(builder_name)
        except KeyError:
            logger.warning(f'Failed to create builder {builder_name}!')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Scan builder "{builder.get_name()}" does not have parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_probe_positions_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builder_factory.create_from_settings()
        except KeyError:
            logger.warning('Failed to create builder from settings!')
            return

        item.set_builder(builder)

    def get_open_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_open_file_filters()

    def get_open_file_filter(self) -> str:
        return self._builder_factory.get_open_file_filter()

    def open_probe_positions(
        self, index: int, file_path: Path, *, file_type: str | None = None
    ) -> None:
        builder = self._builder_factory.create_scan_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open scan {index}!')
        else:
            item.set_builder(builder)

    def copy_scan(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source scan {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination scan {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def get_save_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_save_file_filters()

    def get_save_file_filter(self) -> str:
        return self._builder_factory.get_save_file_filter()

    def save_scan(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save scan {index}!')
        else:
            self._builder_factory.save_scan(file_path, file_type, item.get_probe_positions())


class ProbeAPI:
    def __init__(
        self,
        settings: ProbeSettings,
        repository: ProbeRepository,
        builder_factory: ProbeBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_probe(
        self,
        index: int,
        builder_name: str,
        builder_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create(builder_name, dataset=dataset)
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder {builder_name}: {exc}')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Probe builder "{builder.get_name()}" does not have'
                        f' parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_probe_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create_from_settings(dataset=dataset)
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder from settings: {exc}')
            return

        item.set_builder(builder)

    def get_open_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_open_file_filters()

    def get_open_file_filter(self) -> str:
        return self._builder_factory.get_open_file_filter()

    def open_probe(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
        builder = self._builder_factory.create_probe_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.set_builder(builder)

    def copy_probe(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source probe {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination probe {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def get_save_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_save_file_filters()

    def get_save_file_filter(self) -> str:
        return self._builder_factory.get_save_file_filter()

    def save_probe(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._builder_factory.save_probe(file_path, file_type, item.get_probes())


class ObjectAPI:
    def __init__(
        self,
        settings: ObjectSettings,
        repository: ObjectRepository,
        builder_factory: ObjectBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_object(
        self,
        index: int,
        builder_name: str,
        builder_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create(builder_name, dataset=dataset)
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder {builder_name}: {exc}')
            return

        if builder_parameters is not None:
            for parameter_name, parameter_value in builder_parameters.items():
                try:
                    parameter = builder.parameters()[parameter_name]
                except KeyError:
                    logger.warning(
                        f'Object builder "{builder.get_name()}" does not have'
                        f' parameter "{parameter_name}"!'
                    )
                else:
                    parameter.set_value(parameter_value)

        item.set_builder(builder)

    def build_object_from_settings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        dataset = self._repository.get_dataset(index)

        try:
            builder = self._builder_factory.create_from_settings(dataset=dataset)
        except (KeyError, RuntimeError) as exc:
            logger.warning(f'Failed to create builder from settings: {exc}')
            return

        item.set_builder(builder)

    def get_open_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_open_file_filters()

    def get_open_file_filter(self) -> str:
        return self._builder_factory.get_open_file_filter()

    def open_object(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
        builder = self._builder_factory.create_object_from_file(
            file_path,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.set_builder(builder)

    def copy_object(self, source_index: int, destination_index: int) -> None:
        logger.debug(f'Copying {source_index} -> {destination_index}')

        try:
            source_item = self._repository[source_index]
        except IndexError:
            logger.warning(f'Failed to access source object {source_index} for copying!')
            return

        try:
            destination_item = self._repository[destination_index]
        except IndexError:
            logger.warning(f'Failed to access destination object {destination_index} for copying!')
            return

        destination_item.assign_item(source_item)

    def get_save_file_filters(self) -> Iterator[str]:
        return self._builder_factory.get_save_file_filters()

    def get_save_file_filter(self) -> str:
        return self._builder_factory.get_save_file_filter()

    def save_object(self, index: int, file_path: Path, file_type: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._builder_factory.save_object(file_path, file_type, item.get_object())


class ProductAPI:
    def __init__(
        self,
        settings: ProductSettings,
        repository: ProductRepository,
        item_factory: ProductRepositoryItemFactory,
        file_reader_chooser: PluginChooser[ProductFileReader],
        file_writer_chooser: PluginChooser[ProductFileWriter],
        task_manager: TaskManager,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._item_factory = item_factory
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._task_manager = task_manager

    def _insert_via_queue(
        self,
        dataset: AssembledDiffractionDataset | None,
        build: Callable[[], ProductRepositoryItem],
        *,
        block: bool,
        stub_name: str,
    ) -> int:
        """If dataset is None or already loaded, run build() synchronously and insert.
        Otherwise, insert a pending stub, then enqueue construction to run after the
        dataset's LoadAllArrays finishes (via the shared FIFO background worker)."""
        if dataset is None or not dataset.is_load_in_progress():
            item = build()
            return self._repository.insert_product(item)

        finished_event = dataset.get_last_load_finished_event()

        if block:
            if finished_event is not None:
                while not self._task_manager.is_stopping:
                    if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                        break
            error = dataset.get_last_load_error()
            if error is not None:
                raise RuntimeError('Diffraction dataset failed to load') from error
            item = build()
            return self._repository.insert_product(item)

        stub = self._item_factory.create_pending_stub(name=stub_name)
        index = self._repository.insert_product(stub)

        def background_finalize() -> Callable[[], None]:
            error = dataset.get_last_load_error()

            def foreground_finalize() -> None:
                if error is not None:
                    logger.error(
                        f'Cancelling queued product {index} because dataset '
                        f'failed to load: {error!r}'
                    )
                    stub.set_state(ProductState.FAILED)
                    return

                try:
                    real = build()
                except Exception:
                    logger.exception(f'Queued product {index} construction failed')
                    stub.set_state(ProductState.FAILED)
                    return

                stub.copy_contents_from(real)
                stub.set_state(ProductState.READY)

            return foreground_finalize

        self._task_manager.put_background_task(background_finalize)
        return index

    def insert_new_product(
        self,
        name: str = 'Unnamed',
        *,
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
        tomography_angle_deg: float | None = None,
        tilt_angle_deg: float | None = None,
        polarization: Polarization | None = None,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_values(
                name=name,
                comments=comments,
                detector_distance_m=detector_distance_m,
                probe_energy_eV=probe_energy_eV,
                probe_photon_count=probe_photon_count,
                exposure_time_s=exposure_time_s,
                mass_attenuation_m2_kg=mass_attenuation_m2_kg,
                tomography_angle_deg=tomography_angle_deg,
                tilt_angle_deg=tilt_angle_deg,
                polarization=polarization,
                dataset=dataset,
            )

        return self._insert_via_queue(dataset, build, block=block, stub_name=name)

    def insert_product(
        self,
        product: Product,
        *,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_product(product, dataset=dataset)

        return self._insert_via_queue(dataset, build, block=block, stub_name=product.metadata.name)

    def insert_product_from_settings(
        self,
        *,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        def build() -> ProductRepositoryItem:
            return self._item_factory.create_from_settings(dataset=dataset)

        return self._insert_via_queue(dataset, build, block=block, stub_name='Unnamed')

    def get_item(self, product_index: int) -> ProductRepositoryItem:
        return self._repository[product_index]

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def open_product(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        dataset: AssembledDiffractionDataset | None = None,
        block: bool = True,
    ) -> int:
        if file_path.is_file():
            if file_type is not None:
                self._file_reader_chooser.set_current_plugin(file_type)

            file_type = self._file_reader_chooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{file_path}" as "{file_type}"')
            file_reader = self._file_reader_chooser.get_current_plugin().strategy

            try:
                product = file_reader.read(file_path)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{file_path}"') from exc
            else:
                return self.insert_product(product, dataset=dataset, block=block)
        else:
            logger.warning(f'Refusing to create product with invalid file path "{file_path}"')

        return -1

    def rename_product(self, product_index: int, new_name: str) -> None:
        try:
            item = self._repository[product_index]
        except IndexError:
            logger.warning(f'Failed to access product {product_index} for renaming!')
            return

        item.set_name(new_name)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_product(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save product {index}!')
            return

        if file_type is not None:
            self._file_writer_chooser.set_current_plugin(file_type)

        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        writer = self._file_writer_chooser.get_current_plugin().strategy
        writer.write(file_path, item.get_product())
