from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional
import logging
import threading

import h5py
import numpy
import watchdog.events
import watchdog.observers

from .chooser import StrategyChooser, StrategyEntry
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup
from .tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class DataSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', None)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DataSettings:
        settings = cls(settingsRegistry.createGroup('Data'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class H5FileEventHandler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self) -> None:
        super().__init__(patterns=['*.h5', '*.hdf5'],
                         ignore_directories=True,
                         case_sensitive=False)

    def on_any_event(self, event) -> None:
        print(f'{event.event_type}: {event.src_path}')


class DataDirectoryWatcher(threading.Thread):
    def __init__(self) -> None:
        super().__init__()
        self._directoryPath: Path = Path.home()
        self._observer = watchdog.observers.Observer()
        self._eventHandler = H5FileEventHandler()
        self._waitTimeInSeconds = 1.
        self._stopEvent = threading.Event()
        self._watch = None

    def run(self) -> None:
        self._observer.schedule(self._eventHandler, self._directoryPath, recursive=False)
        self._observer.start()

        try:
            while not self._stopEvent.wait(self._waitTimeInSeconds):
                pass

        except:
            self._observer.stop()
            logger.error('Watchdog Thread Error')  # TODO improve message

        self._observer.join()

    def stop(self) -> None:
        self._observer.stop()
        self._stopEvent.set()


class DatasetState(Enum):
    MISSING = auto()
    FOUND = auto()
    VALID = auto()


@dataclass(frozen=True)
class DataFile:
    name: str
    filePath: Path
    dataPath: str

    def getState(self) -> DatasetState:
        state = DatasetState.MISSING

        if self.filePath.is_file():
            state = DatasetState.FOUND

            try:
                with h5py.File(self.filePath, 'r') as h5File:
                    if self.dataPath in h5File:
                        state = DatasetState.VALID
            except OSError:
                pass

        return state


class DataFileReader(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def getFileContentsTree(self) -> SimpleTreeNode:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> None:
        pass


class DataFilePresenter(Observable, Observer):
    @staticmethod
    def _createFileReaderEntry(fileReader: DataFileReader) -> StrategyEntry[DataFileReader]:
        return StrategyEntry(simpleName=fileReader.simpleName,
                             displayName=fileReader.fileFilter,
                             strategy=fileReader)

    def __init__(self, settings: DataSettings, fileReaderList: list[DataFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._fileReaderChooser = StrategyChooser[DataFileReader].createFromList([
            DataFilePresenter._createFileReaderEntry(fileReader) for fileReader in fileReaderList
        ])
        self._filePath: Optional[Path] = None

    @classmethod
    def createInstance(cls, settings: DataSettings,
                       fileReaderList: list[DataFileReader]) -> DataFilePresenter:
        presenter = cls(settings, fileReaderList)
        settings.fileType.addObserver(presenter)
        presenter._fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()

        settings.filePath.addObserver(presenter)
        presenter._openDataFileFromSettings()

        return presenter

    def getFileContentsTree(self) -> SimpleTreeNode:
        fileReader = self._fileReaderChooser.getCurrentStrategy()
        return fileReader.getFileContentsTree()

    def openDataSet(self, dataPath: str) -> Any:
        data = None

        if self._filePath and dataPath:
            with h5py.File(self._filePath, 'r') as h5File:
                if dataPath in h5File:
                    item = h5File.get(dataPath)

                    if isinstance(item, h5py.Dataset):
                        data = item[()]
                else:
                    parentPath, attrName = dataPath.rsplit('/', 1)

                    if parentPath in h5File:
                        item = h5File.get(parentPath)

                        if attrName in item.attrs:
                            attr = item.attrs[attrName]
                            stringInfo = h5py.check_string_dtype(attr.dtype)

                            if stringInfo:
                                data = attr.decode(stringInfo.encoding)
                            else:
                                data = attr

        return data

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.fileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.fileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openDataFile(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            self._filePath = filePath
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            fileReader.read(filePath)
            self.notifyObservers()

    def openDataFile(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.filePath.value == filePath:
            self._openDataFile(filePath)

        self._settings.filePath.value = filePath

    def _openDataFileFromSettings(self) -> None:
        self._openDataFile(self._settings.filePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDataFileFromSettings()
