from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import ProductFileReader, ProductFileWriter

from .object.builder_factory import ObjectBuilderFactory
from .object.settings import ObjectSettings
from .object_repository import ObjectRepository
from .probe.builder_factory import ProbeBuilderFactory
from .probe.settings import ProbeSettings
from .probe_repository import ProbeRepository
from .product_repository import ProductRepository
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
        pass  # FIXME

    def append_positions_x(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_x_m.extend(values_m)
        self._triggers_x.extend(trigger_counts)

    def append_positions_y(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_y_m.extend(values_m)
        self._triggers_y.extend(trigger_counts)

    def stop(self) -> None:
        pass  # FIXME


class ScanAPI:
    def __init__(
        self,
        settings: ScanSettings,
        repository: ScanRepository,
        builderFactory: ScanBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builderFactory = builderFactory

    def create_streaming_context(self) -> PositionsStreamingContext:
        return PositionsStreamingContext()

    def builder_names(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def buildScan(
        self, index: int, builderName: str, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder.parameters()[parameterName]
            except KeyError:
                logger.warning(
                    f'Scan builder "{builder.get_name()}" does not have parameter "{parameterName}"!'
                )
            else:
                parameter.set_value(parameterValue)

        item.set_builder(builder)

    def buildScanFromSettings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create_from_settings()
        except KeyError:
            logger.warning('Failed to create builder from settings!')
            return

        item.set_builder(builder)

    def get_open_file_filters(self) -> Iterator[str]:
        return self._builderFactory.get_open_file_filters()

    def get_open_file_filter(self) -> str:
        return self._builderFactory.get_open_file_filter()

    def open_scan(self, index: int, filePath: Path, *, file_type: str | None = None) -> None:
        builder = self._builderFactory.create_scan_from_file(
            filePath,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open scan {index}!')
        else:
            item.set_builder(builder)

    def copyScan(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source scan {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination scan {destinationIndex} for copying!')
            return

        destinationItem.assign_item(sourceItem)

    def get_save_file_filters(self) -> Iterator[str]:
        return self._builderFactory.get_save_file_filters()

    def get_save_file_filter(self) -> str:
        return self._builderFactory.get_save_file_filter()

    def save_scan(self, index: int, filePath: Path, fileType: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save scan {index}!')
        else:
            self._builderFactory.save_scan(filePath, fileType, item.get_scan())


class ProbeAPI:
    def __init__(
        self,
        settings: ProbeSettings,
        repository: ProbeRepository,
        builderFactory: ProbeBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builderFactory = builderFactory

    def builderNames(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def buildProbe(
        self, index: int, builderName: str, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder.parameters()[parameterName]
            except KeyError:
                logger.warning(
                    f'Probe builder "{builder.get_name()}" does not have'
                    f' parameter "{parameterName}"!'
                )
            else:
                parameter.set_value(parameterValue)

        item.set_builder(builder)

    def buildProbeFromSettings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create_from_settings()
        except KeyError:
            logger.warning('Failed to create builder from settings!')
            return

        item.set_builder(builder)

    def getOpenFileFilterList(self) -> Iterator[str]:
        return self._builderFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._builderFactory.getOpenFileFilter()

    def openProbe(self, index: int, filePath: Path, *, file_type: str | None = None) -> None:
        builder = self._builderFactory.createProbeFromFile(
            filePath,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.set_builder(builder)

    def copyProbe(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source probe {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination probe {destinationIndex} for copying!')
            return

        destinationItem.assign_item(sourceItem)

    def getSaveFileFilterList(self) -> Iterator[str]:
        return self._builderFactory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._builderFactory.getSaveFileFilter()

    def saveProbe(self, index: int, filePath: Path, fileType: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._builderFactory.saveProbe(filePath, fileType, item.get_probe())


class ObjectAPI:
    def __init__(
        self,
        settings: ObjectSettings,
        repository: ObjectRepository,
        builderFactory: ObjectBuilderFactory,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._builderFactory = builderFactory

    def builderNames(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def build_object(
        self, index: int, builderName: str, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder.parameters()[parameterName]
            except KeyError:
                logger.warning(
                    f'Object builder "{builder.get_name()}" does not have'
                    f' parameter "{parameterName}"!'
                )
            else:
                parameter.set_value(parameterValue)

        item.set_builder(builder)

    def buildObjectFromSettings(self, index: int) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.createFromSettings()
        except KeyError:
            logger.warning('Failed to create builder from settings!')
            return

        item.set_builder(builder)

    def getOpenFileFilterList(self) -> Iterator[str]:
        return self._builderFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._builderFactory.getOpenFileFilter()

    def openObject(self, index: int, filePath: Path, *, file_type: str | None = None) -> None:
        builder = self._builderFactory.createObjectFromFile(
            filePath,
            self._settings.file_type.get_value() if file_type is None else file_type,
        )

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.set_builder(builder)

    def copyObject(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source object {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination object {destinationIndex} for copying!')
            return

        destinationItem.assign_item(sourceItem)

    def getSaveFileFilterList(self) -> Iterator[str]:
        return self._builderFactory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._builderFactory.getSaveFileFilter()

    def saveObject(self, index: int, filePath: Path, fileType: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._builderFactory.saveObject(filePath, fileType, item.get_object())


class ProductAPI:
    def __init__(
        self,
        settings: ProductSettings,
        repository: ProductRepository,
        fileReaderChooser: PluginChooser[ProductFileReader],
        fileWriterChooser: PluginChooser[ProductFileWriter],
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    def insert_new_product(
        self,
        name: str = 'Unnamed',
        *,
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
        like_index: int = -1,
    ) -> int:
        return self._repository.insert_new_product(
            name=name,
            comments=comments,
            detector_distance_m=detectorDistanceInMeters,
            probe_energy_eV=probeEnergyInElectronVolts,
            probe_photon_count=probePhotonCount,
            exposure_time_s=exposureTimeInSeconds,
            like_index=like_index,
        )

    def getItemName(self, productIndex: int) -> str:
        item = self._repository[productIndex]
        return item.get_name()

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._fileReaderChooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._fileReaderChooser.get_current_plugin().display_name

    def open_product(self, filePath: Path, *, file_type: str | None = None) -> int:
        if filePath.is_file():
            if file_type is not None:
                self._fileReaderChooser.set_current_plugin(file_type)

            file_type = self._fileReaderChooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{filePath}" as "{file_type}"')
            fileReader = self._fileReaderChooser.get_current_plugin().strategy

            try:
                product = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                return self._repository.insert_product(product)
        else:
            logger.warning(f'Refusing to create product with invalid file path "{filePath}"')

        return -1

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._fileWriterChooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._fileWriterChooser.get_current_plugin().display_name

    def save_product(self, index: int, filePath: Path, *, file_type: str | None = None) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save product {index}!')
            return

        if file_type is not None:
            self._fileWriterChooser.set_current_plugin(file_type)

        file_type = self._fileWriterChooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{filePath}" as "{file_type}"')
        writer = self._fileWriterChooser.get_current_plugin().strategy
        writer.write(filePath, item.get_product())
