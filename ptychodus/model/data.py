from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Any, Optional, Union
import logging
import tempfile
import threading

import h5py
import numpy
import watchdog.events
import watchdog.observers

from ..api.data import DatasetState, DataArrayType, DataFile, DataFileReader, DiffractionDataset
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser
from ..api.settings import SettingsRegistry, SettingsGroup
from ..api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class DataSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.scratchDirectory = settingsGroup.createPathEntry('ScratchDirectory',
                                                              Path(tempfile.gettempdir()))

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

        except Exception:
            logger.exception('Watchdog Thread Exception!')
            self._observer.stop()

        self._observer.join()

    def stop(self) -> None:
        self._observer.stop()
        self._stopEvent.set()


class NullDiffractionDataset(DiffractionDataset):

    @property
    def datasetName(self) -> str:
        return str()

    @property
    def datasetState(self) -> DatasetState:
        return DatasetState.NOT_FOUND

    def getArray(self) -> DataArrayType:
        return numpy.empty((0, 0, 0), dtype=int)

    @overload
    def __getitem__(self, index: int) -> DataArrayType:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DataArrayType]:
        ...

    def __getitem__(self, index: Union[int,
                                       slice]) -> Union[DataArrayType, Sequence[DataArrayType]]:

        if isinstance(index, slice):
            return list()
        else:
            return numpy.empty((0, 0), dtype=int)

    def __len__(self) -> int:
        return 0


class NullDataFile(DataFile):

    def __init__(self) -> None:
        super().__init__()

    @property
    def metadata(self) -> DataFileMetadata:
        return DataFileMetadata(Path('/dev/null'), 0, 0, 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return SimpleTreeNode(None, list())

    @overload
    def __getitem__(self, index: int) -> DiffractionDataset:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionDataset]:
        ...

    def __getitem__(
            self, index: Union[int,
                               slice]) -> Union[DiffractionDataset, Sequence[DiffractionDataset]]:
        if isinstance(index, slice):
            return list()
        else:
            return NullDiffractionDataset()

    def __len__(self) -> int:
        return 0


class ActiveDataFile(DataFile):

    def __init__(self, settings: DataSettings) -> None:
        super().__init__()
        self._settings = settings
        self._dataFile: DataFile = NullDataFile()
        self._datasetList: list[DiffractionDataset] = list()
        self._dataArray = numpy.empty((0, 0, 0), dtype=int)

    @property
    def metadata(self) -> DataFileMetadata:
        return self._dataFile.metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._dataFile.getContentsTree()

    @overload
    def __getitem__(self, index: int) -> DiffractionDataset:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionDataset]:
        ...

    def __getitem__(
            self, index: Union[int,
                               slice]) -> Union[DiffractionDataset, Sequence[DiffractionDataset]]:
        return self._datasetList[index]

    def __len__(self) -> int:
        return len(self._datasetList)

    def setActive(self, dataFile: DataFile) -> None:
        self._dataFile = dataFile
        self._datasetList.clear()
        self._dataArray = numpy.empty((0, 0, 0), dtype=int)

        npyTempFile = tempfile.NamedTemporaryFile(dir=self._settings.scratchDirectory.value,
                                                  suffix='.npy')
        datasetShape = (self.totalNumberOfImages, self.imageHeight, self.imageWidth)
        logger.debug(f'Scratch data file {npyTempFile.name} is {datasetShape}')
        self._dataArray = numpy.memmap(npyTempFile, dtype=int, shape=datasetShape)

        # FIXME dst[:] = src[:]

        self.notifyObservers()


class DataFilePresenter(Observable, Observer):

    def __init__(self, settings: DataSettings, activeDataFile: ActiveDataFile,
                 fileReaderChooser: PluginChooser[DataFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDataFile = activeDataFile
        self._fileReaderChooser = fileReaderChooser
        self._filePath: Optional[Path] = None

    @classmethod
    def createInstance(cls, settings: DataSettings, activeDataFile: ActiveDataFile,
                       fileReaderChooser: PluginChooser[DataFileReader]) -> DataFilePresenter:
        presenter = cls(settings, activeDataFile, fileReaderChooser)
        settings.fileType.addObserver(presenter)
        presenter._fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()

        settings.filePath.addObserver(presenter)
        presenter._openDataFileFromSettings()

        activeDataFile.addObserver(presenter)

        return presenter

    def getContentsTree(self) -> SimpleTreeNode:
        contentsTree = self._activeDataFile.getContentsTree()
        return contentsTree

    def getDatasetName(self, index: int) -> str:
        return self._activeDataFile[index].datasetName

    def getDatasetState(self, index: int) -> DatasetState:
        return self._activeDataFile[index].datasetState

    def getNumberOfDatasets(self) -> int:
        return len(self._activeDataFile)

    def openDataset(self, dataPath: str) -> Any:  # TODO hdf5-only
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

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.fileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.fileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openDataFile(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            self._filePath = filePath
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            dataFile = fileReader.read(filePath)
            self._activeDataFile.setActive(dataFile)

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
        elif observable is self._activeDataFile:
            self.notifyObservers()
