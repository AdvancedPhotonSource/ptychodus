from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import overload, Any, Optional, Union
import concurrent.futures
import functools
import logging
import tempfile
import threading

import h5py
import numpy
import watchdog.events
import watchdog.observers

from ..api.data import DatasetState, DataArrayType, DataFile, DataFileReader, DataFileMetadata, DiffractionDataset
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser
from ..api.settings import SettingsRegistry, SettingsGroup
from ..api.tree import SimpleTreeNode
from .geometry import Interval

logger = logging.getLogger(__name__)


class DetectorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numberOfPixelsX = settingsGroup.createIntegerEntry('NumberOfPixelsX', 1024)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.numberOfPixelsY = settingsGroup.createIntegerEntry('NumberOfPixelsY', 1024)
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DetectorSettings:
        settings = cls(settingsRegistry.createGroup('Detector'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class DataSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.scratchDirectory = settingsGroup.createPathEntry('ScratchDirectory',
                                                              Path(tempfile.gettempdir()))
        self.numberOfDataThreads = settingsGroup.createIntegerEntry('NumberOfDataThreads', 8)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DataSettings:
        settings = cls(settingsRegistry.createGroup('Data'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class CropSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', True)
        self.centerXInPixels = settingsGroup.createIntegerEntry('CenterXInPixels', 32)
        self.centerYInPixels = settingsGroup.createIntegerEntry('CenterYInPixels', 32)
        self.extentXInPixels = settingsGroup.createIntegerEntry('ExtentXInPixels', 64)
        self.extentYInPixels = settingsGroup.createIntegerEntry('ExtentYInPixels', 64)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> CropSettings:
        settings = cls(settingsRegistry.createGroup('Crop'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Detector(Observable, Observer):

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> Detector:
        detector = cls(settings)
        settings.addObserver(detector)
        return detector

    def getNumberOfPixelsX(self) -> int:
        return self._settings.numberOfPixelsX.value

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    def getNumberOfPixelsY(self) -> int:
        return self._settings.numberOfPixelsY.value

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class CropSizer(Observer, Observable):

    def __init__(self, settings: CropSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector

    @classmethod
    def createInstance(cls, settings: CropSettings, detector: Detector) -> CropSizer:
        sizer = cls(settings, detector)
        settings.addObserver(sizer)
        detector.addObserver(sizer)
        return sizer

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.value

    def getExtentXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsX())

    def getExtentXInPixels(self) -> int:
        limitsInPixels = self.getExtentXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.extentXInPixels.value)

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentXInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsX() - 1 - radiusInPixels)

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.centerXInPixels.value)

    def getSliceX(self) -> slice:
        centerInPixels = self.getCenterXInPixels()
        radiusInPixels = self.getExtentXInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def getExtentYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsY())

    def getExtentYInPixels(self) -> int:
        limitsInPixels = self.getExtentYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.extentYInPixels.value)

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentYInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsY() - 1 - radiusInPixels)

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.centerYInPixels.value)

    def getSliceY(self) -> slice:
        centerInPixels = self.getCenterYInPixels()
        radiusInPixels = self.getExtentYInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._detector:
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


class CachedDiffractionDataset(DiffractionDataset, Observer):

    def __init__(self, name: str, state: DatasetState, array: DataArrayType,
                 cropSizer: CropSizer) -> None:
        super().__init__()
        self._datasetName = name
        self._datasetState = state
        self._array = array
        self._cropSizer = cropSizer

    @classmethod
    def createInstance(cls, name: str, state: DatasetState, array: DataArrayType,
                       cropSizer: CropSizer) -> CachedDiffractionDataset:
        dataset = cls(name, state, array, cropSizer)
        cropSizer.addObserver(dataset)
        return dataset

    @property
    def datasetName(self) -> str:
        return self._datasetName

    @property
    def datasetState(self) -> DatasetState:
        return self._datasetState

    def getArray(self) -> DataArrayType:
        if self._cropSizer.isCropEnabled():
            sliceX = self._cropSizer.getSliceX()
            sliceY = self._cropSizer.getSliceY()
            return self._array[:, sliceY, sliceX]

        return self._array

    def __getitem__(self, index: int) -> DataArrayType:
        array = numpy.empty((0, 0), dtype=int)

        if self._cropSizer.isCropEnabled():
            sliceX = self._cropSizer.getSliceX()
            sliceY = self._cropSizer.getSliceY()

            array = self._array[index, sliceY, sliceX]
        else:
            array = self._array[index, :, :]

        return array

    def __len__(self) -> int:
        return self._array.shape[0]

    def update(self, observable: Observable) -> None:
        if observable is self._cropSizer:
            self.notifyObservers()


class NullDataFile(DataFile):

    def __init__(self) -> None:
        super().__init__()

    @property
    def metadata(self) -> DataFileMetadata:
        return DataFileMetadata(Path('/dev/null'), 0, 0, 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return SimpleTreeNode.createRoot(list())

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

    def __init__(self, settings: DataSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._cropSizer = cropSizer
        self._dataFile: DataFile = NullDataFile()
        self._datasetList: list[DiffractionDataset] = list()
        self._dataArray = numpy.empty((0, 0, 0), dtype=int)

    @property
    def metadata(self) -> DataFileMetadata:
        return self._dataFile.metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._dataFile.getContentsTree()

    def getDiffractionData(self) -> DataArrayType:
        if self._cropSizer.isCropEnabled():
            sliceX = self._cropSizer.getSliceX()
            sliceY = self._cropSizer.getSliceY()
            return self._dataArray[:, sliceY, sliceX]

        return self._dataArray

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

    def _loadDataset(self, index: int, stride: int,
                     dataset: DiffractionDataset) -> DiffractionDataset:
        logger.debug(f'Reading {dataset.datasetName}...')
        array = dataset.getArray()

        if array.size > 0:
            offset = index * stride
            sliceZ = slice(offset, offset + array.shape[0])
            self._dataArray[sliceZ, ...] = array[...]

            array_slice = self._dataArray[sliceZ, ...].view()
            array_slice.flags.writeable = False

            cachedDataset = CachedDiffractionDataset.createInstance(dataset.datasetName,
                                                                    DatasetState.VALID,
                                                                    array_slice, self._cropSizer)
            return cachedDataset

        return dataset

    def setActive(self, dataFile: DataFile) -> None:
        self._dataFile = dataFile
        self._datasetList.clear()
        self._dataArray = numpy.empty((0, 0, 0), dtype=int)

        npyTempFile = tempfile.NamedTemporaryFile(dir=self._settings.scratchDirectory.value,
                                                  suffix='.npy')
        datasetShape = (dataFile.metadata.totalNumberOfImages, dataFile.metadata.imageHeight,
                        dataFile.metadata.imageWidth)
        logger.debug(f'Scratch data file {npyTempFile.name} is {datasetShape}')
        self._dataArray = numpy.memmap(npyTempFile, dtype=int, shape=datasetShape)
        maxWorkers = self._settings.numberOfDataThreads.value
        stride = int(dataFile.metadata.totalNumberOfImages) // len(dataFile)

        with concurrent.futures.ThreadPoolExecutor(maxWorkers) as executor:
            futureList = list()

            for index, dataset in enumerate(dataFile):
                datasetLoader = functools.partial(self._loadDataset, index, stride, dataset)
                future = executor.submit(datasetLoader)
                futureList.append(future)

            for future in concurrent.futures.as_completed(futureList):
                dataset = future.result()
                self._datasetList.append(dataset)

        self._datasetList.sort(key=lambda x: x.datasetName)
        self.notifyObservers()


class DataFilePresenter(Observable, Observer):

    def __init__(self, settings: DataSettings, activeDataFile: ActiveDataFile,
                 fileReaderChooser: PluginChooser[DataFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDataFile = activeDataFile
        self._fileReaderChooser = fileReaderChooser

    @classmethod
    def createInstance(cls, settings: DataSettings, activeDataFile: ActiveDataFile,
                       fileReaderChooser: PluginChooser[DataFileReader]) -> DataFilePresenter:
        presenter = cls(settings, activeDataFile, fileReaderChooser)
        settings.fileType.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()
        settings.filePath.addObserver(presenter)
        presenter._openDataFileFromSettings()
        activeDataFile.addObserver(presenter)
        return presenter

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getContentsTree(self) -> SimpleTreeNode:
        return self._activeDataFile.getContentsTree()

    def getDatasetName(self, index: int) -> str:
        return self._activeDataFile[index].datasetName

    def getDatasetState(self, index: int) -> DatasetState:
        return self._activeDataFile[index].datasetState

    def getNumberOfDatasets(self) -> int:
        return len(self._activeDataFile)

    def openDataset(self, dataPath: str) -> Any:  # TODO hdf5-only
        filePath = self._activeDataFile.metadata.filePath
        data = None

        if filePath and dataPath:
            with h5py.File(filePath, 'r') as h5File:
                if dataPath in h5File:
                    item = h5File.get(dataPath)

                    if isinstance(item, h5py.Dataset):
                        data = item[()]  # TODO decode strings as needed
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


class CropPresenter(Observer, Observable):

    def __init__(self, settings: CropSettings, sizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: CropSettings, sizer: CropSizer) -> CropPresenter:
        presenter = cls(settings, sizer)
        sizer.addObserver(presenter)
        return presenter

    def isCropEnabled(self) -> bool:
        return self._sizer.isCropEnabled()

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getMinCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().lower

    def getMaxCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().upper

    def getCenterXInPixels(self) -> int:
        return self._sizer.getCenterXInPixels()

    def setCenterXInPixels(self, value: int) -> None:
        self._settings.centerXInPixels.value = value

    def getMinCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().lower

    def getMaxCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().upper

    def getCenterYInPixels(self) -> int:
        return self._sizer.getCenterYInPixels()

    def setCenterYInPixels(self, value: int) -> None:
        self._settings.centerYInPixels.value = value

    def getMinExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().lower

    def getMaxExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().upper

    def getExtentXInPixels(self) -> int:
        return self._sizer.getExtentXInPixels()

    def setExtentXInPixels(self, value: int) -> None:
        self._settings.extentXInPixels.value = value

    def getMinExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().lower

    def getMaxExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().upper

    def getExtentYInPixels(self) -> int:
        return self._sizer.getExtentYInPixels()

    def setExtentYInPixels(self, value: int) -> None:
        self._settings.extentYInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
