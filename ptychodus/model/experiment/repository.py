from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.experiment import (Experiment, ExperimentFileReader, ExperimentFileWriter,
                               ExperimentMetadata)
from ...api.object import Object
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import Probe
from ...api.scan import Scan
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..patterns import PatternSizer
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .sizer import ExperimentSizer

logger = logging.getLogger(__name__)


class ExperimentRepositoryItemObserver(ABC):

    @abstractmethod
    def handleMetadataChanged(self, item: ExperimentRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleScanChanged(self, item: ExperimentRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleProbeChanged(self, item: ExperimentRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleObjectChanged(self, item: ExperimentRepositoryItem) -> None:
        pass


class ExperimentRepositoryItem(Observer):

    def __init__(self, parent: ExperimentRepositoryItemObserver, privateIndex: int,
                 experiment: Experiment, patternSizer: PatternSizer) -> None:
        super().__init__()
        self._parent = parent
        self._privateIndex = privateIndex
        self._metadata = MetadataRepositoryItem(experiment.metadata)
        self._scan = ScanRepositoryItem(experiment.scan)
        self._sizer = ExperimentSizer(self._metadata, self._scan, patternSizer)
        self._probe = ProbeRepositoryItem(experiment.probe)
        self._object = ObjectRepositoryItem(experiment.object_)

        self._metadata.addObserver(self)
        self._scan.addObserver(self)
        self._probe.addObserver(self)
        self._object.addObserver(self)

    def getMetadata(self) -> MetadataRepositoryItem:
        return self._metadata

    def getScan(self) -> ScanRepositoryItem:
        return self._scan

    def getProbe(self) -> ProbeRepositoryItem:
        return self._probe

    def getObject(self) -> ObjectRepositoryItem:
        return self._object

    def getExperiment(self) -> Experiment:
        return Experiment(
            metadata=self._metadata.getMetadata(),
            scan=self._scan.getScan(),
            probe=self._probe.getProbe(),
            object_=self._object.getObject(),
        )

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self._parent.handleMetadataChanged(self)
        elif observable is self._scan:
            self._parent.handleScanChanged(self)
        elif observable is self._probe:
            self._parent.handleProbeChanged(self)
        elif observable is self._object:
            self._parent.handleObjectChanged(self)


class ExperimentRepositoryObserver(ABC):

    @abstractmethod
    def handleItemInserted(self, index: int, item: ExperimentRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    @abstractmethod
    def handleItemRemoved(self, index: int, item: ExperimentRepositoryItem) -> None:
        pass


class ExperimentRepository(Sequence[ExperimentRepositoryItem], ExperimentRepositoryItemObserver):

    def __init__(self, patternSizer: PatternSizer,
                 fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        super().__init__()
        self._patternSizer = patternSizer
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._itemList: list[ExperimentRepositoryItem] = list()
        self._observerList: list[ExperimentRepositoryObserver] = list()

    @overload
    def __getitem__(self, index: int) -> ExperimentRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ExperimentRepositoryItem]:
        ...

    def __getitem__(
            self,
            index: int | slice) -> ExperimentRepositoryItem | Sequence[ExperimentRepositoryItem]:
        return self._itemList[index]

    def __len__(self) -> int:
        return len(self._itemList)

    def _insertExperiment(self, experiment: Experiment) -> None:
        index = len(self._itemList)
        item = ExperimentRepositoryItem(self, index, experiment, self._patternSizer)
        self._itemList.append(item)

        for observer in self._observerList:
            observer.handleItemInserted(index, item)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openExperiment(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(fileFilter)
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                experiment = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read \"{filePath}\"') from exc
            else:
                self._insertExperiment(experiment)
        else:
            logger.debug(f'Refusing to create experiment with invalid file path \"{filePath}\"')

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveExperiment(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._itemList[index]
        except IndexError:
            logger.debug(f'Failed to save experiment {index}!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, item.getExperiment())

    def addObserver(self, observer: ExperimentRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ExperimentRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def insertExperiment(self, name: str) -> None:
        # FIXME extract to factory; populate defaults from settings/metadata
        experiment = Experiment(
            metadata=ExperimentMetadata(name),
            scan=Scan(),
            probe=Probe(),
            object_=Object(),
        )
        self._insertExperiment(experiment)

    def handleMetadataChanged(self, item: ExperimentRepositoryItem) -> None:
        index = item._privateIndex
        metadata = item.getMetadata()

        for observer in self._observerList:
            observer.handleMetadataChanged(index, metadata)

    def handleScanChanged(self, item: ExperimentRepositoryItem) -> None:
        index = item._privateIndex
        scan = item.getScan()

        for observer in self._observerList:
            observer.handleScanChanged(index, scan)

    def handleProbeChanged(self, item: ExperimentRepositoryItem) -> None:
        index = item._privateIndex
        probe = item.getProbe()

        for observer in self._observerList:
            observer.handleProbeChanged(index, probe)

    def handleObjectChanged(self, item: ExperimentRepositoryItem) -> None:
        index = item._privateIndex
        object_ = item.getObject()

        for observer in self._observerList:
            observer.handleObjectChanged(index, object_)

    def _updateIndexes(self) -> None:
        for index, item in enumerate(self._itemList):
            item._privateIndex = index

    def removeExperiment(self, index: int) -> None:
        try:
            item = self._itemList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove experiment item {index}!')
        else:
            self._updateIndexes()

            for observer in self._observerList:
                observer.handleItemRemoved(index, item)
