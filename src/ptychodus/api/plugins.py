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

from .diffraction import (
    BadPixelsFileReader,
    DiffractionFileReader,
    DiffractionFileWriter,
    PixelMapFileReader,
)
from .files import FileHandler
from .fluorescence import (
    DeconvolutionStrategy,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from .object import ObjectFileReader, ObjectFileWriter, Object
from .observer import Observable
from .probe import ProbeFileReader, ProbeFileWriter, ProbeEnsemble
from .simulate.probe import FresnelZonePlate, KirkpatrickBaezMirrorPair
from .probe_positions import (
    ProbePositionFileReader,
    ProbePositionFileWriter,
    ProbePositionSequence,
)
from .product import ProductFileReader, ProductFileWriter
from .reconstruct import ReconstructorLibrary, TrainingDataFileWriter
from .workflow import FileBasedWorkflow

__all__ = [
    'PluginChooser',
    'PluginRegistry',
]

T = TypeVar('T')

logger = logging.getLogger(__name__)


class _ProductReaderAdapter(FileHandler):
    """Base for the adapters that expose one part of a ProductFileReader's output.

    An adapter reads the same files as the reader it wraps, so it describes itself
    as that reader does rather than carrying a format of its own.
    """

    def __init__(self, reader: ProductFileReader) -> None:
        super().__init__()
        self._reader = reader

    def get_file_description(self) -> str:
        return self._reader.get_file_description()

    def get_file_extensions(self) -> set[str]:
        return self._reader.get_file_extensions()


class ProductProbePositionFileReader(_ProductReaderAdapter, ProbePositionFileReader):
    """Adapter that extracts the probe positions from a ProductFileReader."""

    def read(self, file_path: Path) -> ProbePositionSequence:
        product = self._reader.read(file_path)
        return product.probe_positions


class ProductProbeFileReader(_ProductReaderAdapter, ProbeFileReader):
    """Adapter that extracts the probe sequence from a ProductFileReader."""

    def read(self, file_path: Path) -> ProbeEnsemble:
        product = self._reader.read(file_path)
        return product.probes


class ProductObjectFileReader(_ProductReaderAdapter, ObjectFileReader):
    """Adapter that extracts the object from a ProductFileReader."""

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

    Each plugin carries two names: the human-readable ``display_name`` and the
    ``simple_name`` suitable for persistence. The chooser resolves either form and
    notifies observers whenever the selection moves. It holds no settings of its own
    and never writes one.
    """

    def __init__(self) -> None:
        super().__init__()
        self._registered_plugins: list[Plugin[T]] = list()
        self._current_index = 0

    def stringify_plugin_names(self) -> str:
        """Return a sorted, comma-separated list of registered plugin simple names."""
        return ', '.join(sorted(plugin.simple_name for plugin in self._registered_plugins))

    def register_plugin(
        self, strategy: T, *, display_name: str | None = None, simple_name: str = ''
    ) -> None:
        """Register *strategy* under *display_name*; *simple_name* defaults to a stripped form.

        A FileHandler names itself, so its *display_name* is always its own file
        filter and the two cannot drift apart. Any other strategy must supply one.

        Both mistakes below warn rather than raise, for the reason the duplicate
        simple name does: load_plugins catches only AttributeError, so raising here
        would abort startup over one bad registration.
        """
        if isinstance(strategy, FileHandler):
            # Honoring an override is how a registration drifted from the format it
            # names, which is what deriving the filter exists to prevent.
            if display_name is not None:
                logger.warning(
                    f'Ignoring display name "{display_name}" for '
                    f'{type(strategy).__name__}: a file handler is named by its own '
                    f'file filter, "{strategy.get_file_filter()}".'
                )

            display_name = strategy.get_file_filter()
        elif display_name is None:
            # There is no name to register under, so there is nothing to recover.
            logger.warning(
                f'Skipping {type(strategy).__name__}: it is not a FileHandler, so it '
                'cannot name itself and no display_name was given.'
            )
            return

        if not simple_name:
            simple_name = re.sub(r'\W+', '', display_name)

        # Settings persist the simple name, and lookup returns the first match, so a
        # duplicate silently shadows the later registration and breaks its round trip.
        # Warn rather than raise: load_plugins catches only AttributeError, and plugin
        # loading must never be fatal.
        for plugin in self._registered_plugins:
            if plugin.simple_name.casefold() == simple_name.casefold():
                logger.warning(
                    f'Duplicate plugin simple name "{simple_name}": '
                    f'"{display_name}" will be unreachable by that name '
                    f'because "{plugin.display_name}" already claims it.'
                )
                break

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

    def get_strategy_by_name(self, name: str) -> T:
        """Return the strategy of the plugin matching *name*; raise LookupError otherwise."""
        plugin = self.find_plugin(name)
        if plugin is None:
            raise LookupError(
                f'No plugin matches "{name}". Registered: {self.stringify_plugin_names()}.'
            )
        return plugin.strategy

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


class PluginRegistry:
    """Central collection of PluginChooser instances for every plugin category."""

    def __init__(self) -> None:
        self.bad_pixels_file_readers = PluginChooser[BadPixelsFileReader]()
        self.background_file_readers = PluginChooser[PixelMapFileReader]()
        self.flat_field_file_readers = PluginChooser[PixelMapFileReader]()
        self.diffraction_file_readers = PluginChooser[DiffractionFileReader]()
        self.diffraction_file_writers = PluginChooser[DiffractionFileWriter]()
        self.probe_position_file_readers = PluginChooser[ProbePositionFileReader]()
        self.probe_position_file_writers = PluginChooser[ProbePositionFileWriter]()
        self.fresnel_zone_plates = PluginChooser[FresnelZonePlate]()
        self.kb_mirrors = PluginChooser[KirkpatrickBaezMirrorPair]()
        self.probe_file_readers = PluginChooser[ProbeFileReader]()
        self.probe_file_writers = PluginChooser[ProbeFileWriter]()
        self.object_file_readers = PluginChooser[ObjectFileReader]()
        self.object_file_writers = PluginChooser[ObjectFileWriter]()
        self.product_file_readers = PluginChooser[ProductFileReader]()
        self.product_file_writers = PluginChooser[ProductFileWriter]()
        self.file_based_workflows = PluginChooser[FileBasedWorkflow]()
        self.fluorescence_file_readers = PluginChooser[FluorescenceFileReader]()
        self.fluorescence_file_writers = PluginChooser[FluorescenceFileWriter]()
        self.training_data_file_writers = PluginChooser[TrainingDataFileWriter]()
        self.upscaling_strategies = PluginChooser[UpscalingStrategy]()
        self.deconvolution_strategies = PluginChooser[DeconvolutionStrategy]()
        # A reconstructor backend is discovered the same way a file format is:
        # by registering itself, rather than by being named somewhere. An
        # instance like every other chooser holds, which a backend can supply
        # because it is configured per call and holds nothing of its own.
        self.reconstructor_libraries = PluginChooser[ReconstructorLibrary]()

    def register_product_file_reader_with_adapters(
        self, strategy: ProductFileReader, *, simple_name: str = ''
    ) -> None:
        """Register *strategy* as a product reader, and as probe-position, probe, and object readers via adapters."""
        self.probe_position_file_readers.register_plugin(
            ProductProbePositionFileReader(strategy), simple_name=simple_name
        )
        self.probe_file_readers.register_plugin(
            ProductProbeFileReader(strategy), simple_name=simple_name
        )
        self.object_file_readers.register_plugin(
            ProductObjectFileReader(strategy), simple_name=simple_name
        )
        self.product_file_readers.register_plugin(strategy, simple_name=simple_name)

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
