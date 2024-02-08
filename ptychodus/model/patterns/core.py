from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import h5py
import numpy

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.patterns import DiffractionFileReader, DiffractionPatternArrayType, DiffractionPatternState
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from ...api.state import DiffractionPatternStateData, StatefulCore
from ...api.tree import SimpleTreeNode
from .active import ActiveDiffractionDataset
from .api import DiffractionDataAPI
from .builder import ActiveDiffractionDatasetBuilder
from .detector import Detector, DetectorPresenter
from .io import DiffractionDatasetInputOutputPresenter
from .metadata import DiffractionMetadataPresenter
from .patterns import DiffractionPatternPresenter
from .settings import DiffractionDatasetSettings, DiffractionPatternSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffractionPatternArrayPresenter:
    label: str
    state: DiffractionPatternState
    data: DiffractionPatternArrayType | None

    @classmethod
    def createNull(cls) -> DiffractionPatternArrayPresenter:
        return cls(str(), DiffractionPatternState.UNKNOWN, None)


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, settings: DiffractionDatasetSettings,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset

    @classmethod
    def createInstance(cls, settings: DiffractionDatasetSettings,
                       dataset: ActiveDiffractionDataset) -> DiffractionDatasetPresenter:
        presenter = cls(settings, dataset)
        settings.addObserver(presenter)
        dataset.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[DiffractionPatternArrayPresenter]:
        for array in self._dataset:
            yield DiffractionPatternArrayPresenter(
                label=array.getLabel(),
                state=array.getState(),
                data=array.getData(),
            )

    def __len__(self) -> int:
        return len(self._dataset)

    def getInfoText(self) -> str:
        return self._dataset.getInfoText()

    def isMemmapEnabled(self) -> bool:
        return self._settings.memmapEnabled.value

    def setMemmapEnabled(self, value: bool) -> None:
        self._settings.memmapEnabled.value = value

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getNumberOfDataThreadsLimits(self) -> Interval[int]:
        return Interval[int](1, 64)

    def getNumberOfDataThreads(self) -> int:
        limits = self.getNumberOfDataThreadsLimits()
        return limits.clamp(self._settings.numberOfDataThreads.value)

    def setNumberOfDataThreads(self, number: int) -> None:
        self._settings.numberOfDataThreads.value = number

    @property
    def isAssembled(self) -> bool:
        return (len(self._dataset) > 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return self._dataset.getContentsTree()

    def openArray(self, dataPath: str) -> Any:  # TODO generalize for other file formats
        filePath = self._dataset.getMetadata().filePath
        data = None

        if filePath and h5py.is_hdf5(filePath) and dataPath:
            try:
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
            except OSError:
                logger.exception('Failed to open dataset!')

        return data

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._dataset:
            self.notifyObservers()


class PatternsCore(StatefulCore[DiffractionPatternStateData]):

    def __init__(self, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[DiffractionFileReader]) -> None:
        self.detector = Detector.createInstance(settingsRegistry)
        self.detectorPresenter = DetectorPresenter.createInstance(self.detector)
        self.datasetSettings = DiffractionDatasetSettings.createInstance(settingsRegistry)
        self._patternSettings = DiffractionPatternSettings.createInstance(settingsRegistry)

        # TODO vvv refactor vvv
        fileReaderChooser.setCurrentPluginByName(self.datasetSettings.fileType.value)

        self.patternSizer = PatternSizer.createInstance(self._patternSettings, self.detector)
        self.patternPresenter = DiffractionPatternPresenter.createInstance(
            self._patternSettings, self.patternSizer)

        self.dataset = ActiveDiffractionDataset(self.datasetSettings, self._patternSettings,
                                                self.patternSizer)
        self._builder = ActiveDiffractionDatasetBuilder(self.datasetSettings, self.dataset)
        self.dataAPI = DiffractionDataAPI(self._builder, fileReaderChooser)

        self.metadataPresenter = DiffractionMetadataPresenter(self.dataset, self.detector,
                                                              self.datasetSettings,
                                                              self._patternSettings)
        self.datasetPresenter = DiffractionDatasetPresenter.createInstance(
            self.datasetSettings, self.dataset)
        self.datasetInputOutputPresenter = DiffractionDatasetInputOutputPresenter.createInstance(
            self.datasetSettings, self.dataset, self.dataAPI, settingsRegistry)

    def getStateData(self) -> DiffractionPatternStateData:
        return DiffractionPatternStateData(
            indexes=numpy.array(self.dataset.getAssembledIndexes()),
            array=self.dataset.getAssembledData(),
        )

    def setStateData(self, stateData: DiffractionPatternStateData, stateFilePath: Path) -> None:
        if self._builder.isAssembling:
            self._builder.stop(finishAssembling=False)

        self.dataset.setAssembledData(stateData.array, stateData.indexes)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._builder.stop(finishAssembling=False)
