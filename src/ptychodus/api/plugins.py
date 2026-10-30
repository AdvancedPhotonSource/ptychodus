"""Plugin registry and per-category catalogs for managing ptychodus extensions.

Plugins are discovered at runtime by :meth:`PluginRegistry.load_plugins`, which
walks ``ptychodus.plugins.*`` and calls each module's ``register_plugins(registry)``
function. Modules whose imports fail (typically optional-dependency plugins) are
logged and skipped so that missing dependencies silently disable plugins rather
than crashing the application.
"""

from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Generic, TypeVar, overload
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
    FluorescenceEnhancer,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)
from .object import ObjectFileReader, ObjectFileWriter, Object
from .probe import ProbeFileReader, ProbeFileWriter, ProbeEnsemble
from .simulate.object import ObjectGenerator
from .simulate.probe import FresnelZonePlate, KirkpatrickBaezMirrorPreset, ProbeGenerator
from .simulate.probe_positions import ProbePositionsGenerator
from .probe_positions import (
    ProbePositionFileReader,
    ProbePositionFileWriter,
    ProbePositionSequence,
)
from .product import ProductFileReader, ProductFileWriter
from .reconstruct import ReconstructorLibrary, TrainingDataFileWriter
from .workflow import FileBasedWorkflow

__all__ = [
    'Plugin',
    'PluginCatalog',
    'PluginList',
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


class PluginCatalog(Sequence[Plugin[T]], ABC):
    """Read-only view of the plugins registered in one category, sorted by display name.

    Each plugin carries two names: the human-readable ``display_name`` and the
    ``simple_name`` suitable for persistence. Lookup resolves either form,
    case-insensitively. A catalog holds no selection and notifies no one.
    """

    def stringify_plugin_names(self) -> str:
        """Return a sorted, comma-separated list of registered plugin simple names."""
        return ', '.join(sorted(plugin.simple_name for plugin in self))

    def find_plugin(self, name: str) -> Plugin[T] | None:
        """Return the plugin matching *name* (case-insensitive simple or display name), or None."""
        namecf = name.casefold()

        for plugin in self:
            if namecf == plugin.simple_name.casefold() or namecf == plugin.display_name.casefold():
                return plugin

        return None

    def get_strategy_by_name(self, name: str) -> T:
        """Return the strategy of the plugin matching *name*; raise LookupError otherwise."""
        plugin = self.find_plugin(name)
        if plugin is None:
            raise LookupError(
                f'No plugin matches "{name}". Registered: {self.stringify_plugin_names()}.'
            )
        return plugin.strategy


class PluginList(PluginCatalog[T]):
    """A catalog that accepts registrations until it is frozen."""

    def __init__(self) -> None:
        super().__init__()
        self._registered_plugins: list[Plugin[T]] = list()
        self._is_frozen = False

    def freeze(self) -> None:
        """Refuse further registrations, so the catalog is read-only from here on."""
        self._is_frozen = True

    def register_plugin(
        self, strategy: T, *, display_name: str | None = None, simple_name: str = ''
    ) -> None:
        """Register *strategy* under *display_name*; *simple_name* defaults to a stripped form.

        A FileHandler names itself, so its *display_name* is always its own file
        filter and the two cannot drift apart. Any other strategy must supply one.

        Every mistake below warns rather than raises, for the reason the duplicate
        simple name does: load_plugins catches only AttributeError, so raising here
        would abort startup over one bad registration.
        """
        if self._is_frozen:
            logger.warning(
                f'Skipping {type(strategy).__name__}: the catalog is frozen, so it '
                'accepts no registrations after plugin loading.'
            )
            return

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

        self._registered_plugins.append(Plugin[T](strategy, simple_name, display_name))
        self._registered_plugins.sort(key=lambda x: x.display_name)

    @overload
    def __getitem__(self, index: int) -> Plugin[T]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Plugin[T]]: ...

    def __getitem__(self, index: int | slice) -> Plugin[T] | Sequence[Plugin[T]]:
        return self._registered_plugins[index]

    def __len__(self) -> int:
        return len(self._registered_plugins)


class PluginRegistry:
    """Central collection of the plugin catalogs, one per plugin category.

    Plugin modules register into the catalogs while :meth:`load_plugins` runs; once it
    returns, every catalog is frozen and read-only.
    """

    def __init__(self) -> None:
        self.bad_pixels_file_readers = PluginList[BadPixelsFileReader]()
        self.background_file_readers = PluginList[PixelMapFileReader]()
        self.flat_field_file_readers = PluginList[PixelMapFileReader]()
        self.diffraction_file_readers = PluginList[DiffractionFileReader]()
        self.diffraction_file_writers = PluginList[DiffractionFileWriter]()
        self.probe_position_file_readers = PluginList[ProbePositionFileReader]()
        self.probe_position_file_writers = PluginList[ProbePositionFileWriter]()
        # A scan pattern is discovered the same way a file format is, and states
        # what it accepts the same way a reconstructor does. An instance like
        # every other catalog holds, which a generator can supply because it is
        # configured per call and holds nothing of its own.
        self.probe_position_generators = PluginList[ProbePositionsGenerator]()
        # A probe model is offered the way a scan pattern is.
        self.probe_generators = PluginList[ProbeGenerator]()
        self.fresnel_zone_plates = PluginList[FresnelZonePlate]()
        self.kb_mirrors = PluginList[KirkpatrickBaezMirrorPreset]()
        self.probe_file_readers = PluginList[ProbeFileReader]()
        self.probe_file_writers = PluginList[ProbeFileWriter]()
        # An object model is offered the way a probe model is.
        self.object_generators = PluginList[ObjectGenerator]()
        self.object_file_readers = PluginList[ObjectFileReader]()
        self.object_file_writers = PluginList[ObjectFileWriter]()
        self.product_file_readers = PluginList[ProductFileReader]()
        self.product_file_writers = PluginList[ProductFileWriter]()
        self.file_based_workflows = PluginList[FileBasedWorkflow]()
        self.fluorescence_file_readers = PluginList[FluorescenceFileReader]()
        self.fluorescence_file_writers = PluginList[FluorescenceFileWriter]()
        self.training_data_file_writers = PluginList[TrainingDataFileWriter]()
        # A fluorescence enhancer is offered the way an object model is.
        self.fluorescence_enhancers = PluginList[FluorescenceEnhancer]()
        # A reconstructor backend is discovered the same way a file format is:
        # by registering itself, rather than by being named somewhere. An
        # instance like every other catalog holds, which a backend can supply
        # because it is configured per call and holds nothing of its own.
        self.reconstructor_libraries = PluginList[ReconstructorLibrary]()

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
        """Return a frozen registry populated by importing every ``ptychodus.plugins.*`` module and calling its ``register_plugins`` hook."""
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

        for catalog in vars(registry).values():
            if isinstance(catalog, PluginList):
                catalog.freeze()

        return registry
