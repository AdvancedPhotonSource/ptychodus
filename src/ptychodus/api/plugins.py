from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, TypeVar
import importlib
import logging
import pkgutil
import re

from .fluorescence import (
    DeconvolutionStrategy,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from .object import ObjectFileReader, ObjectFileWriter
from .observer import Observable, Observer
from .parametric import StringParameter
from .patterns import DiffractionFileReader, DiffractionFileWriter
from .probe import FresnelZonePlate, ProbeFileReader, ProbeFileWriter
from .product import ProductFileReader, ProductFileWriter
from .scan import ScanFileReader, ScanFileWriter
from .workflow import FileBasedWorkflow

__all__ = [
    'PluginChooser',
    'PluginRegistry',
]

T = TypeVar('T')

logger = logging.getLogger(__name__)


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

        logger.debug(f'Invalid plugin name "{name}"')

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
        self.diffraction_file_readers = PluginChooser[DiffractionFileReader]()
        self.diffraction_file_writers = PluginChooser[DiffractionFileWriter]()
        self.scan_file_readers = PluginChooser[ScanFileReader]()
        self.scan_file_writers = PluginChooser[ScanFileWriter]()
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
                logger.info(f'Registering {module_info.name}')
                module.register_plugins(registry)

        return registry
