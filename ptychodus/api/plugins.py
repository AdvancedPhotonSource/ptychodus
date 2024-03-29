from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, TypeVar
import importlib
import logging
import pkgutil

from .data import DiffractionFileReader
from .image import ScalarTransformation
from .object import ObjectPhaseCenteringStrategy, ObjectFileReader, ObjectFileWriter
from .observer import Observable
from .probe import ProbeFileReader, ProbeFileWriter
from .scan import ScanFileReader, ScanFileWriter

__all__ = [
    'PluginEntry',
    'PluginChooser',
    'PluginRegistry',
]

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginEntry(Generic[T]):
    strategy: T
    simpleName: str
    displayName: str


class PluginChooser(Generic[T], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._entryList: list[PluginEntry[T]] = list()
        self._currentIndex = 0

    def getSimpleNameList(self) -> Sequence[str]:
        return [entry.simpleName for entry in self._entryList]

    def getDisplayNameList(self) -> Sequence[str]:
        return [entry.displayName for entry in self._entryList]

    def registerPlugin(self, strategy: T, *, simpleName: str, displayName: str = '') -> None:
        if not displayName:
            displayName = simpleName

        entry = PluginEntry[T](strategy, simpleName, displayName)
        self._entryList.append(entry)
        self.notifyObservers()

    @property
    def currentPlugin(self) -> PluginEntry[T]:
        return self._entryList[self._currentIndex]

    def setCurrentPluginByName(self, name: str) -> None:
        namecf = name.casefold()

        for index, entry in enumerate(self._entryList):
            if namecf == entry.simpleName.casefold() or namecf == entry.displayName.casefold():
                if self._currentIndex != index:
                    self._currentIndex = index
                    self.notifyObservers()

                return

        logger.debug(f'Invalid plugin name \"{name}\"')

    def __iter__(self) -> Iterator[PluginEntry[T]]:
        return iter(self._entryList)

    def __getitem__(self, name: str) -> PluginEntry[T]:
        namecf = name.casefold()

        for entry in self._entryList:
            if namecf == entry.simpleName.casefold() or namecf == entry.displayName.casefold():
                return entry

        raise KeyError(f'Invalid plugin name \"{name}\"')

    def __bool__(self) -> bool:
        return bool(self._entryList)

    def copy(self) -> PluginChooser[T]:
        clone = PluginChooser[T]()
        clone._entryList = self._entryList.copy()
        clone._currentIndex = self._currentIndex
        return clone


class PluginRegistry:

    def __init__(self) -> None:
        self.diffractionFileReaders = PluginChooser[DiffractionFileReader]()
        self.scalarTransformations = PluginChooser[ScalarTransformation]()
        self.scanFileReaders = PluginChooser[ScanFileReader]()
        self.scanFileWriters = PluginChooser[ScanFileWriter]()
        self.probeFileReaders = PluginChooser[ProbeFileReader]()
        self.probeFileWriters = PluginChooser[ProbeFileWriter]()
        self.objectPhaseCenteringStrategies = PluginChooser[ObjectPhaseCenteringStrategy]()
        self.objectFileReaders = PluginChooser[ObjectFileReader]()
        self.objectFileWriters = PluginChooser[ObjectFileWriter]()

    @classmethod
    def loadPlugins(cls) -> PluginRegistry:
        registry = cls()

        import ptychodus.plugins
        ns_pkg: ModuleType = ptychodus.plugins

        # Specifying the second argument (prefix) to iter_modules makes the
        # returned name an absolute name instead of a relative one. This allows
        # import_module to work without having to do additional modification to
        # the name.
        for moduleInfo in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + '.'):
            logger.info(f'Importing {moduleInfo.name}')
            module = importlib.import_module(moduleInfo.name)
            module.registerPlugins(registry)

        return registry
