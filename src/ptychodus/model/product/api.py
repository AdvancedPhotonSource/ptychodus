from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter

from .item import ProductRepositoryItem
from .item_factory import ProductRepositoryItemFactory
from .object.builder_factory import ObjectBuilderFactory
from .object.settings import ObjectSettings
from .object_repository import ObjectRepository
from .probe.builder_factory import ProbeBuilderFactory
from .probe.settings import ProbeSettings
from .probe_repository import ProbeRepository
from .repository import ProductRepository
from .scan.builder_factory import ScanBuilderFactory
from .scan.settings import ScanSettings
from .scan_repository import ScanRepository
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


class ScanAPI:
    def __init__(
        self,
        settings: ScanSettings,
        repository: ScanRepository,
        builder_factory: ScanBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builder_factory = builder_factory

    def create_streaming_context(self) -> PositionsStreamingContext:
        return PositionsStreamingContext()

    def builder_names(self) -> Iterator[str]:
        return iter(self._builder_factory)

    def build_scan(
        self, index: int, builder_name: str, builder_parameters: Mapping[str, Any] = {}
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

    def build_scan_from_settings(self, index: int) -> None:
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

    def open_scan(self, index: int, file_path: Path, *, file_type: str | None = None) -> None:
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
            self._builder_factory.save_scan(file_path, file_type, item.get_scan())


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
        self, index: int, builder_name: str, builder_parameters: Mapping[str, Any] = {}
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
        self, index: int, builder_name: str, builder_parameters: Mapping[str, Any] = {}
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
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._item_factory = item_factory
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser

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
    ) -> int:
        item = self._item_factory.create_from_values(
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
        )
        return self._repository.insert_product(item)

    def insert_product(self, product: Product) -> int:
        item = self._item_factory.create_from_product(product)
        return self._repository.insert_product(item)

    def insert_product_from_settings(self) -> int:
        item = self._item_factory.create_from_settings()
        return self._repository.insert_product(item)

    def get_item(self, product_index: int) -> ProductRepositoryItem:
        return self._repository[product_index]

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def open_product(self, file_path: Path, *, file_type: str | None = None) -> int:
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
                item = self._item_factory.create_from_product(product)
                return self._repository.insert_product(item)
        else:
            logger.warning(f'Refusing to create product with invalid file path "{file_path}"')

        return -1

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
