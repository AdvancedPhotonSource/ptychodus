"""Plugin registry and chooser for managing ptychodus extensions.

Plugins are discovered at runtime by :meth:`PluginRegistry.load_plugins`, which
walks ``ptychodus.plugins.*`` and calls each module's ``register_plugins(registry)``
function. Modules whose imports fail (typically optional-dependency plugins) are
logged and skipped so that missing dependencies silently disable plugins rather
than crashing the application.
"""

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
from .parametric import Parameter, StringParameter
from .probe import ProbeFileReader, ProbeFileWriter, ProbeSequence
from .probe_gen import FresnelZonePlate
from .probe_positions import (
    ProbePositionFileReader,
    ProbePositionFileWriter,
    ProbePositionSequence,
)
from .product import ProductFileReader, ProductFileWriter
from .workflow import FileBasedWorkflow

__all__ = [
    'PluginChooser',
    'PluginChooserParameter',
    'PluginRegistry',
]

T = TypeVar('T')

logger = logging.getLogger(__name__)


class ProductProbePositionFileReader(ProbePositionFileReader):
    """Adapter that extracts the probe positions from a ProductFileReader."""

    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> ProbePositionSequence:
        product = self._reader.read(file_path)
        return product.probe_positions


class ProductProbeFileReader(ProbeFileReader):
    """Adapter that extracts the probe sequence from a ProductFileReader."""

    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> ProbeSequence:
        product = self._reader.read(file_path)
        return product.probes


class ProductObjectFileReader(ObjectFileReader):
    """Adapter that extracts the object from a ProductFileReader."""

    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def read(self, file_path: Path) -> Object:
        product = self._reader.read(file_path)
        return product.object_


@dataclass(frozen=True)
class Plugin(Generic[T]):
    """A registered plugin: its strategy object and both simple and display names."""

    strategy: T
    simple_name: str
    display_name: str


class PluginChooser(Iterable[Plugin[T]], Observable):
    """Observable list of typed plugins with a tracked current selection.

    The chooser knows nothing about settings persistence. Bind it to a settings
    parameter by constructing a :class:`PluginChooserParameter` over it, which owns
    the translation between the two name spaces in both directions.
    """

    def __init__(self) -> None:
        super().__init__()
        self._registered_plugins: list[Plugin[T]] = list()
        self._current_index = 0

    def stringify_plugin_names(self) -> str:
        """Return a sorted, comma-separated list of registered plugin simple names."""
        return ', '.join(sorted(plugin.simple_name for plugin in self._registered_plugins))

    def register_plugin(self, strategy: T, *, display_name: str, simple_name: str = '') -> None:
        """Register *strategy* under *display_name*; *simple_name* defaults to a stripped form."""
        if not simple_name:
            simple_name = re.sub(r'\W+', '', display_name)

        # The list is kept sorted by display name, so a registration can move the
        # selected plugin to a different index. Track it by identity across the sort
        # rather than by value: Plugin is a frozen dataclass, so equality compares the
        # strategy field and an array-valued strategy would break the comparison.
        current = (
            self._registered_plugins[self._current_index] if self._registered_plugins else None
        )

        self._registered_plugins.append(Plugin[T](strategy, simple_name, display_name))
        self._registered_plugins.sort(key=lambda x: x.display_name)

        if current is not None:
            self._current_index = next(
                index for index, plugin in enumerate(self._registered_plugins) if plugin is current
            )

        self.notify_observers()

    def _find_index(self, name: str) -> int | None:
        namecf = name.casefold()

        for index, plugin in enumerate(self._registered_plugins):
            if namecf == plugin.simple_name.casefold() or namecf == plugin.display_name.casefold():
                return index

        return None

    def find_plugin(self, name: str) -> Plugin[T] | None:
        """Return the plugin matching *name* (case-insensitive simple or display name), or None."""
        index = self._find_index(name)
        return None if index is None else self._registered_plugins[index]

    def get_current_plugin(self) -> Plugin[T]:
        """Return the currently selected plugin."""
        if not self._registered_plugins:
            raise LookupError('No plugins registered')
        return self._registered_plugins[self._current_index]

    def set_current_plugin(self, name: str) -> None:
        """Select the plugin matching *name* (case-insensitive simple or display name).

        An unrecognized name logs a warning and leaves the selection unchanged, but
        still notifies observers so that a bound view resynchronizes to the selection
        the chooser actually holds.
        """
        index = self._find_index(name)

        if index is None:
            registered_plugins = ', '.join(
                f'"{plugin.simple_name}"' for plugin in self._registered_plugins
            )
            logger.warning(
                f'Invalid plugin name "{name}". Registered plugins: {registered_plugins}.'
            )
            self.notify_observers()
            return

        if index != self._current_index:
            self._current_index = index
            self.notify_observers()

    def __iter__(self) -> Iterator[Plugin[T]]:
        for plugin in self._registered_plugins:
            yield plugin

    def __bool__(self) -> bool:
        return bool(self._registered_plugins)


class PluginChooserParameter(Parameter[str], Observer, Generic[T]):
    """Parameter[str] view of a PluginChooser whose value space is plugin display names.

    A chooser has two name spaces: the human-readable ``display_name`` shown in the
    GUI and the ``simple_name`` persisted to settings. This adapter is the single
    place that translates between them. Its own value space is display names, so a
    plain combo box bound to it round trips correctly.

    Passing *settings* additionally binds the chooser to that settings parameter: the
    persisted name selects a plugin at construction and is rewritten to its canonical
    simple name, and every later selection change writes the simple name back. A
    persisted name that matches no registered plugin is left alone rather than
    reconciled, because a plugin can be missing merely because its optional
    dependency failed to import this run.

    Note that :meth:`get_value` raises ``LookupError`` when no plugins are registered,
    so this must not be read before ``PluginRegistry.load_plugins()`` has run.
    """

    def __init__(self, chooser: PluginChooser[T], settings: StringParameter | None = None) -> None:
        super().__init__()
        self._chooser = chooser
        self._settings = settings
        self._selected: Plugin[T] | None = None
        self._suppress_notify = False

        if settings is not None:
            self._apply_settings()
            settings.add_observer(self)

        self._selected = chooser.get_current_plugin() if chooser else None
        chooser.add_observer(self)

    def choices(self) -> Iterator[str]:
        """Yield the display names to populate a combo box with."""
        for plugin in self._chooser:
            yield plugin.display_name

    def get_strategy(self) -> T:
        return self._chooser.get_current_plugin().strategy

    def get_value(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def set_value(self, value: str, *, notify: bool = True) -> None:
        # The chooser notifies us back through _update; suppress that relay rather
        # than notifying here, so a no-op selection stays silent either way. Only the
        # relay is suppressed: persistence must not depend on a view flag.
        self._suppress_notify = not notify

        try:
            self._chooser.set_current_plugin(value)
        finally:
            self._suppress_notify = False

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def copy(self) -> Parameter[str]:
        """Return an unbound view of the same chooser; the copy does not persist."""
        return PluginChooserParameter(self._chooser)

    def _apply_settings(self) -> None:
        settings = self._settings

        if settings is None:
            return

        name = settings.get_value()
        plugin = self._chooser.find_plugin(name)
        self._chooser.set_current_plugin(name)

        if plugin is not None:
            # Unconditional, so that a display name normalizes to its simple name even
            # when it already resolves to the current selection. ParameterBase guards
            # against no-op writes, so this does not notify unless the value changed.
            settings.set_value(plugin.simple_name)

    def _reconcile(self) -> None:
        """Settle the chooser and the settings parameter against each other.

        Observable notifications carry no payload, so the cached selection is what
        distinguishes "the selection moved" from "a plugin was registered". Only the
        former may write back; the latter must leave an unresolved setting intact.
        """
        settings = self._settings
        plugin = self._chooser.get_current_plugin() if self._chooser else None

        if plugin is None or settings is None:
            self._selected = plugin
            return

        if self._selected is None:
            # The chooser just gained its first plugin: the selection came into
            # existence rather than moving, so the fallback must not be persisted.
            self._selected = plugin

        if plugin is not self._selected:
            self._selected = plugin
            settings.set_value(plugin.simple_name)
            return

        # The selection held, so this was a registration. If it has just made the
        # persisted name resolvable, honor it now — set_current_plugin re-enters
        # here through the chooser's notification to finish the write-back.
        desired = self._chooser.find_plugin(settings.get_value())

        if desired is not None and desired is not plugin:
            self._apply_settings()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._apply_settings()
        elif observable is self._chooser:
            self._reconcile()

            if not self._suppress_notify:
                self.notify_observers()


class PluginRegistry:
    """Central collection of PluginChooser instances for every plugin category."""

    def __init__(self) -> None:
        self.bad_pixels_file_readers = PluginChooser[BadPixelsFileReader]()
        self.diffraction_file_readers = PluginChooser[DiffractionFileReader]()
        self.diffraction_file_writers = PluginChooser[DiffractionFileWriter]()
        self.probe_position_file_readers = PluginChooser[ProbePositionFileReader]()
        self.probe_position_file_writers = PluginChooser[ProbePositionFileWriter]()
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
        """Register *strategy* as a product reader, and as probe-position, probe, and object readers via adapters."""
        self.probe_position_file_readers.register_plugin(
            ProductProbePositionFileReader(strategy),
            display_name=display_name,
            simple_name=simple_name,
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
        """Return a registry populated by importing every ``ptychodus.plugins.*`` module and calling its ``register_plugins`` hook."""
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
                logger.warning(exc)
                logger.warning(f'Skipping {module_info.name}')
            else:
                try:
                    module.register_plugins(registry)
                except AttributeError as exc:
                    logger.warning(exc)
                    logger.warning(f'Failed to register {module_info.name}')
                else:
                    logger.debug(f'Registered {module_info.name}')

        return registry
