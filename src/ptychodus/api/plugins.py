from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Generic, TypeVar
import importlib
import logging
import pkgutil
import re

from .diffraction import BadPixelsFileReader, DiffractionFileReader, DiffractionFileWriter
from .fluorescence import (
    DeconvolutionStrategy,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from .object import ObjectFileReader, ObjectFileWriter, Object
from .observer import Observable, Observer
from .parametric import StringParameter
from .probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter, ProbeSequence
from .product import ProductFileReader, ProductFileWriter
from .scan import PositionFileReader, PositionFileWriter, PositionSequence
from .workflow import FileBasedWorkflow

__all__ = [
    'PluginChooser',
    'PluginRegistry',
]

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ProductPositionFileReader(PositionFileReader):
    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> PositionSequence:
        product = self._reader.read(file_path)
        return product.positions


class ProductProbeFileReader(ProbeFileReader):
    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> ProbeSequence:
        product = self._reader.read(file_path)
        return product.probes


class ProductObjectFileReader(ObjectFileReader):
    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> Object:
        product = self._reader.read(file_path)
        return product.object_


@dataclass(frozen=True)
class Plugin(Generic[T]):
    strategy: T
    simple_name: str
    display_name: str


class PluginChooser(Iterable[Plugin[T]], Observable, Observer):
    def __init__(self) -> None:
        super().__init__()
        self._registered_plugins: list[Plugin[T]] = list()
        self._current_index = 0
        self._parameter: StringParameter | None = None

    def register_plugin(self, strategy: T, *, display_name: str, simple_name: str = '') -> None:
        if not simple_name:
            simple_name = re.sub(r'\W+', '', display_name)

        plugin = Plugin[T](strategy, simple_name, display_name)
        self._registered_plugins.append(plugin)
        self._registered_plugins.sort(key=lambda x: x.display_name)
        self.notify_observers()

    def get_current_plugin(self) -> Plugin[T]:
        return self._registered_plugins[self._current_index]

    def set_current_plugin(self, name: str) -> None:
        namecf = name.casefold()

        for index, plugin in enumerate(self._registered_plugins):
            if namecf == plugin.simple_name.casefold() or namecf == plugin.display_name.casefold():
                if self._current_index != index:
                    self._current_index = index

                    if self._parameter is not None:
                        self._parameter.set_value(self.get_current_plugin().simple_name)

                    self.notify_observers()

                return

        registered_plugins = ', '.join(f'"{pi.simple_name}"' for pi in self._registered_plugins)
        logger.debug(f'Invalid plugin name "{name}". Registered plugins: {registered_plugins}.')

    def synchronize_with_parameter(self, parameter: StringParameter) -> None:
        self._parameter = parameter
        self.set_current_plugin(parameter.get_value())
        self._parameter.add_observer(self)

    def __iter__(self) -> Iterator[Plugin[T]]:
        for plugin in self._registered_plugins:
            yield plugin

    def __bool__(self) -> bool:
        return bool(self._registered_plugins)

    def _update(self, observable: Observable) -> None:
        if self._parameter is not None and observable is self._parameter:
            self.set_current_plugin(self._parameter.get_value())


class PluginRegistry:
    def __init__(self) -> None:
        self.bad_pixels_file_readers = PluginChooser[BadPixelsFileReader]()
        self.diffraction_file_readers = PluginChooser[DiffractionFileReader]()
        self.diffraction_file_writers = PluginChooser[DiffractionFileWriter]()
        self.position_file_readers = PluginChooser[PositionFileReader]()
        self.position_file_writers = PluginChooser[PositionFileWriter]()
        self.fresnel_zone_plates = PluginChooser[FresnelZonePlate]()
        self.probe_file_readers = PluginChooser[ProbeFileReader]()
        self.probe_file_writers = PluginChooser[ProbeFileWriter]()
        self.object_file_readers = PluginChooser[ObjectFileReader]()
        self.object_file_writers = PluginChooser[ObjectFileWriter]()
        self.product_file_readers = PluginChooser[ProductFileReader]()
        self.product_file_writers = PluginChooser[ProductFileWriter]()
        self.file_based_workflows = PluginChooser[FileBasedWorkflow]()
        self.fluorescence_file_readers = PluginChooser[FluorescenceFileReader]()
        self.fluorescence_file_writers = PluginChooser[FluorescenceFileWriter]()
        self.upscaling_strategies = PluginChooser[UpscalingStrategy]()
        self.deconvolution_strategies = PluginChooser[DeconvolutionStrategy]()

    def register_product_file_reader_with_adapters(
        self, strategy: ProductFileReader, *, display_name: str, simple_name: str = ''
    ) -> None:
        self.position_file_readers.register_plugin(
            ProductPositionFileReader(strategy), display_name=display_name, simple_name=simple_name
        )
        self.probe_file_readers.register_plugin(
            ProductProbeFileReader(strategy), display_name=display_name, simple_name=simple_name
        )
        self.object_file_readers.register_plugin(
            ProductObjectFileReader(strategy), display_name=display_name, simple_name=simple_name
        )
        self.product_file_readers.register_plugin(
            strategy, display_name=display_name, simple_name=simple_name
        )

    @classmethod
    def load_plugins(cls) -> PluginRegistry:
        registry = cls()

        import ptychodus.plugins

        ns_pkg: ModuleType = ptychodus.plugins

        # Specifying the second argument (prefix) to iter_modules makes the
        # returned name an absolute name instead of a relative one. This allows
        # import_module to work without having to do additional modification to
        # the name.
        for module_info in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + '.'):
            try:
                module = importlib.import_module(module_info.name)
            except ModuleNotFoundError as exc:
                logger.info(f'Skipping {module_info.name}')
                logger.warning(exc)
            else:
                try:
                    module.register_plugins(registry)
                except AttributeError as exc:
                    logger.info(f'Failed to register {module_info.name}')
                    logger.warning(exc)
                else:
                    logger.info(f'Registered {module_info.name}')

        return registry
