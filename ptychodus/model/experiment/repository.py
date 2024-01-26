from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload
import logging

from ...api.experiment import Experiment
from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ..metadata import MetadataRepositoryItem
from ..object import ObjectRepositoryItem, ObjectRepositoryItemFactory
from ..patterns import ActiveDiffractionDataset, PatternSizer
from ..probe import ProbeRepositoryItem, ProbeRepositoryItemFactory
from ..scan import ScanRepositoryItem, ScanRepositoryItemFactory
from .geometry import ExperimentGeometry

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


class ExperimentRepositoryItem(ParameterRepository):

    def __init__(self, parent: ExperimentRepositoryItemObserver, privateIndex: int,
                 metadata: MetadataRepositoryItem, scan: ScanRepositoryItem,
                 geometry: ExperimentGeometry, probe: ProbeRepositoryItem,
                 object_: ObjectRepositoryItem, patterns: ActiveDiffractionDataset) -> None:
        super().__init__('Experiment')
        self._parent = parent
        self._privateIndex = privateIndex
        self._metadata = metadata
        self._scan = scan
        self._probe = probe
        self._object = object_
        self._geometry = geometry
        self._patterns = patterns

        self._addParameterRepository(self._metadata)
        self._addParameterRepository(self._scan)
        self._addParameterRepository(self._probe)
        self._addParameterRepository(self._object)
        self._addParameterRepository(self._geometry)

        self._metadata.addObserver(self)
        self._scan.addObserver(self)
        self._probe.addObserver(self)
        self._object.addObserver(self)
        self._patterns.addObserver(self)

    def getMetadata(self) -> MetadataRepositoryItem:
        return self._metadata

    def getScan(self) -> ScanRepositoryItem:
        return self._scan

    def isScanValid(self) -> bool:
        # FIXME self.notifyObservers() when isScanValid changes
        scan = self._scan.getScan()
        scanIndexes = set(point.index for point in scan)
        patternIndexes = set(self._patterns.getAssembledIndexes())
        return (not scanIndexes.isdisjoint(patternIndexes))

    def getProbe(self) -> ProbeRepositoryItem:
        return self._probe

    def isProbeValid(self) -> bool:
        # FIXME self.notifyObservers() when isProbeValid changes
        probe = self._probe.getProbe()
        return self._geometry.isProbeGeometryValid(probe.getGeometry())

    def getObject(self) -> ObjectRepositoryItem:
        return self._object

    def isObjectValid(self) -> bool:
        # FIXME self.notifyObservers() when isObjectValid changes
        object_ = self._object.getObject()
        return self._geometry.isObjectGeometryValid(object_.getGeometry())

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
        elif observable is self._patterns:
            self.notifyObservers()


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

    def __init__(self, patternSizer: PatternSizer, patterns: ActiveDiffractionDataset,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory) -> None:
        super().__init__()
        self._patternSizer = patternSizer
        self._patterns = patterns
        self._scanRepositoryItemFactory = scanRepositoryItemFactory
        self._probeRepositoryItemFactory = probeRepositoryItemFactory
        self._objectRepositoryItemFactory = objectRepositoryItemFactory
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

    def insertExperiment(self, experiment: Experiment) -> int:
        index = len(self._itemList)

        metadata = MetadataRepositoryItem(experiment.metadata)
        scan = self._scanRepositoryItemFactory.create(experiment.scan)
        geometry = ExperimentGeometry(metadata, scan, self._patternSizer)

        item = ExperimentRepositoryItem(
            parent=self,
            privateIndex=index,
            metadata=metadata,
            scan=scan,
            geometry=geometry,
            probe=self._probeRepositoryItemFactory.create(experiment.probe),
            object_=self._objectRepositoryItemFactory.create(experiment.object_),
            patterns=self._patterns)
        self._itemList.append(item)

        for observer in self._observerList:
            observer.handleItemInserted(index, item)

        return index

    def addObserver(self, observer: ExperimentRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ExperimentRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

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
