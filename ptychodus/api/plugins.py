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
    simpleName: str
    displayName: str
    strategy: T


class PluginChooser(Generic[T], Observable):

    def __init__(self, defaultEntry: PluginEntry[T]) -> None:
        super().__init__()
        self._entryList: list[PluginEntry[T]] = [defaultEntry]
        self._entry: PluginEntry[T] = defaultEntry

    @classmethod
    def createFromList(cls, entryList: list[PluginEntry[T]]) -> PluginChooser[T]:
        chooser = cls(entryList[0])
        chooser._entryList = entryList.copy()
        return chooser

    def addStrategy(self, entry: PluginEntry[T]) -> None:
        self._entryList.insert(0, entry)

    def setToDefault(self) -> None:
        self._setEntry(self._entryList[-1])

    def getSimpleNameList(self) -> Sequence[str]:
        return [entry.simpleName for entry in self._entryList]

    def setFromSimpleName(self, name: str) -> None:
        try:
            entry = next(entry for entry in self._entryList
                         if name.casefold() == entry.simpleName.casefold())
        except StopIteration:
            logger.debug(f'Invalid strategy simple name \"{name}\"')
            return

        self._setEntry(entry)

    def getDisplayNameList(self) -> Sequence[str]:
        return [entry.displayName for entry in self._entryList]

    def setFromDisplayName(self, name: str) -> None:
        try:
            entry = next(entry for entry in self._entryList if name == entry.displayName)
        except StopIteration:
            logger.debug(f'Invalid strategy display name \"{name}\"')
            return

        self._setEntry(entry)

    def getCurrentSimpleName(self) -> str:
        return self._entry.simpleName

    def getCurrentDisplayName(self) -> str:
        return self._entry.displayName

    def getCurrentStrategy(self) -> T:
        return self._entry.strategy

    def _setEntry(self, entry: PluginEntry[T]) -> None:
        if self._entry is not entry:
            self._entry = entry
            self.notifyObservers()

    def __iter__(self) -> Iterator[PluginEntry[T]]:
        return iter(self._entryList)

    def __getitem__(self, index: int) -> PluginEntry[T]:
        return self._entryList[index]

    def __len__(self) -> int:
        return len(self._entryList)


class PluginRegistry:

    def __init__(self) -> None:
        self.diffractionFileReaders: list[PluginEntry[DiffractionFileReader]] = list()
        self.scalarTransformations: list[PluginEntry[ScalarTransformation]] = list()
        self.scanFileReaders: list[PluginEntry[ScanFileReader]] = list()
        self.scanFileWriters: list[PluginEntry[ScanFileWriter]] = list()
        self.probeFileReaders: list[PluginEntry[ProbeFileReader]] = list()
        self.probeFileWriters: list[PluginEntry[ProbeFileWriter]] = list()
        self.objectPhaseCenteringStrategies: list[
            PluginEntry[ObjectPhaseCenteringStrategy]] = list()
        self.objectFileReaders: list[PluginEntry[ObjectFileReader]] = list()
        self.objectFileWriters: list[PluginEntry[ObjectFileWriter]] = list()

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

    def registerPlugin(self, plugin: T) -> None:
        if isinstance(plugin, DiffractionFileReader):
            diffractionFileReaderEntry = PluginEntry[DiffractionFileReader](
                simpleName=plugin.simpleName, displayName=plugin.fileFilter, strategy=plugin)
            self.diffractionFileReaders.append(diffractionFileReaderEntry)
        elif isinstance(plugin, ScalarTransformation):
            scalarTransformationEntry = PluginEntry[ScalarTransformation](simpleName=plugin.name,
                                                                          displayName=plugin.name,
                                                                          strategy=plugin)
            self.scalarTransformations.append(scalarTransformationEntry)
        elif isinstance(plugin, ScanFileReader):
            scanFileReaderEntry = PluginEntry[ScanFileReader](simpleName=plugin.simpleName,
                                                              displayName=plugin.fileFilter,
                                                              strategy=plugin)
            self.scanFileReaders.append(scanFileReaderEntry)
        elif isinstance(plugin, ScanFileWriter):
            scanFileWriterEntry = PluginEntry[ScanFileWriter](simpleName=plugin.simpleName,
                                                              displayName=plugin.fileFilter,
                                                              strategy=plugin)
            self.scanFileWriters.append(scanFileWriterEntry)
        elif isinstance(plugin, ProbeFileReader):
            probeFileReaderEntry = PluginEntry[ProbeFileReader](simpleName=plugin.simpleName,
                                                                displayName=plugin.fileFilter,
                                                                strategy=plugin)
            self.probeFileReaders.append(probeFileReaderEntry)
        elif isinstance(plugin, ProbeFileWriter):
            probeFileWriterEntry = PluginEntry[ProbeFileWriter](simpleName=plugin.simpleName,
                                                                displayName=plugin.fileFilter,
                                                                strategy=plugin)
            self.probeFileWriters.append(probeFileWriterEntry)
        elif isinstance(plugin, ObjectPhaseCenteringStrategy):
            objectPhaseCenteringStrategyEntry = PluginEntry[ObjectPhaseCenteringStrategy](
                simpleName=plugin.name, displayName=plugin.name, strategy=plugin)
            self.objectPhaseCenteringStrategies.append(objectPhaseCenteringStrategyEntry)
        elif isinstance(plugin, ObjectFileReader):
            objectFileReaderEntry = PluginEntry[ObjectFileReader](simpleName=plugin.simpleName,
                                                                  displayName=plugin.fileFilter,
                                                                  strategy=plugin)
            self.objectFileReaders.append(objectFileReaderEntry)
        elif isinstance(plugin, ObjectFileWriter):
            objectFileWriterEntry = PluginEntry[ObjectFileWriter](simpleName=plugin.simpleName,
                                                                  displayName=plugin.fileFilter,
                                                                  strategy=plugin)
            self.objectFileWriters.append(objectFileWriterEntry)
        else:
            raise TypeError(f'Invalid plugin type \"{type(plugin).__name__}\".')

    def buildDiffractionFileReaderChooser(self) -> PluginChooser[DiffractionFileReader]:
        return PluginChooser[DiffractionFileReader].createFromList(self.diffractionFileReaders)

    def buildScalarTransformationChooser(self) -> PluginChooser[ScalarTransformation]:
        return PluginChooser[ScalarTransformation].createFromList(self.scalarTransformations)

    def buildScanFileReaderChooser(self) -> PluginChooser[ScanFileReader]:
        return PluginChooser[ScanFileReader].createFromList(self.scanFileReaders)

    def buildScanFileWriterChooser(self) -> PluginChooser[ScanFileWriter]:
        return PluginChooser[ScanFileWriter].createFromList(self.scanFileWriters)

    def buildProbeFileReaderChooser(self) -> PluginChooser[ProbeFileReader]:
        return PluginChooser[ProbeFileReader].createFromList(self.probeFileReaders)

    def buildProbeFileWriterChooser(self) -> PluginChooser[ProbeFileWriter]:
        return PluginChooser[ProbeFileWriter].createFromList(self.probeFileWriters)

    def buildObjectPhaseCenteringStrategyChooser(
            self) -> PluginChooser[ObjectPhaseCenteringStrategy]:
        return PluginChooser[ObjectPhaseCenteringStrategy].createFromList(
            self.objectPhaseCenteringStrategies)

    def buildObjectFileReaderChooser(self) -> PluginChooser[ObjectFileReader]:
        return PluginChooser[ObjectFileReader].createFromList(self.objectFileReaders)

    def buildObjectFileWriterChooser(self) -> PluginChooser[ObjectFileWriter]:
        return PluginChooser[ObjectFileWriter].createFromList(self.objectFileWriters)
