from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, TypeVar, overload
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
from .object import ObjectPhaseCenteringStrategy, ObjectFileReader, ObjectFileWriter
from .observer import Observable
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
class PluginEntry(Generic[T]):
    strategy: T
    simpleName: str
    displayName: str


class PluginChooser(Sequence[PluginEntry[T]], Observable):
    def __init__(self) -> None:
        super().__init__()
        self._entryList: list[PluginEntry[T]] = list()
        self._currentIndex = 0

    def getSimpleNameList(self) -> Sequence[str]:
        return [entry.simpleName for entry in self._entryList]

    def getDisplayNameList(self) -> Sequence[str]:
        return [entry.displayName for entry in self._entryList]

    def registerPlugin(self, strategy: T, *, displayName: str, simpleName: str = '') -> None:
        if not simpleName:
            simpleName = re.sub(r'\W+', '', displayName)

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

        logger.debug(f'Invalid plugin name "{name}"')

    @overload
    def __getitem__(self, index: int) -> PluginEntry[T]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[PluginEntry[T]]: ...

    def __getitem__(self, index: int | slice) -> PluginEntry[T] | Sequence[PluginEntry[T]]:
        return self._entryList[index]

    def __len__(self) -> int:
        return len(self._entryList)

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
        self.diffractionFileWriters = PluginChooser[DiffractionFileWriter]()
        self.scanFileReaders = PluginChooser[ScanFileReader]()
        self.scanFileWriters = PluginChooser[ScanFileWriter]()
        self.fresnelZonePlates = PluginChooser[FresnelZonePlate]()
        self.probeFileReaders = PluginChooser[ProbeFileReader]()
        self.probeFileWriters = PluginChooser[ProbeFileWriter]()
        self.objectPhaseCenteringStrategies = PluginChooser[ObjectPhaseCenteringStrategy]()
        self.objectFileReaders = PluginChooser[ObjectFileReader]()
        self.objectFileWriters = PluginChooser[ObjectFileWriter]()
        self.productFileReaders = PluginChooser[ProductFileReader]()
        self.productFileWriters = PluginChooser[ProductFileWriter]()
        self.fileBasedWorkflows = PluginChooser[FileBasedWorkflow]()
        self.fluorescenceFileReaders = PluginChooser[FluorescenceFileReader]()
        self.fluorescenceFileWriters = PluginChooser[FluorescenceFileWriter]()
        self.upscalingStrategies = PluginChooser[UpscalingStrategy]()
        self.deconvolutionStrategies = PluginChooser[DeconvolutionStrategy]()

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
            try:
                module = importlib.import_module(moduleInfo.name)
            except ModuleNotFoundError as exc:
                logger.info(f'Skipping {moduleInfo.name}')
                logger.warning(exc)
            else:
                logger.info(f'Registering {moduleInfo.name}')
                module.registerPlugins(registry)

        return registry
