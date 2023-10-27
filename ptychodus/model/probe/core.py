from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import Probe, ProbeArrayType, ProbeFileReader, ProbeFileWriter
from ...api.settings import SettingsRegistry
from ...api.state import ProbeStateData, StatefulCore
from ..data import DiffractionPatternSizer
from ..detector import Detector
from .api import ProbeAPI
from .apparatus import Apparatus, ApparatusPresenter
from .factory import ProbeRepositoryItemFactory
from .modes import MultimodalProbeFactory
from .repository import ProbeRepository, ProbeRepositoryItem
from .selected import ProbeRepositoryItemSettingsDelegate, SelectedProbe
from .settings import ProbeSettings
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeRepositoryItemPresenter:
    name: str
    item: ProbeRepositoryItem


class ProbeRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ProbeRepository, itemFactory: ProbeRepositoryItemFactory,
                 probeAPI: ProbeAPI, fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._itemFactory = itemFactory
        self._probeAPI = probeAPI
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(
            cls, repository: ProbeRepository, itemFactory: ProbeRepositoryItemFactory,
            probeAPI: ProbeAPI,
            fileWriterChooser: PluginChooser[ProbeFileWriter]) -> ProbeRepositoryPresenter:
        presenter = cls(repository, itemFactory, probeAPI, fileWriterChooser)
        repository.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[ProbeRepositoryItemPresenter]:
        for name, item in self._repository.items():
            yield ProbeRepositoryItemPresenter(name, item)

    def __getitem__(self, index: int) -> ProbeRepositoryItemPresenter:
        nameItemTuple = self._repository.getNameItemTupleByIndex(index)
        return ProbeRepositoryItemPresenter(*nameItemTuple)

    def __len__(self) -> int:
        return len(self._repository)

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._itemFactory.getInitializerDisplayNameList()

    def initializeProbe(self, displayName: str) -> str | None:
        return self._probeAPI.insertItemIntoRepositoryFromInitializerName(displayName)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._probeAPI.insertItemIntoRepositoryFromFile(filePath, fileFilter)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveProbe(self, name: str, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, item.getProbe())

        if item.getInitializer() is None:
            initializer = self._itemFactory.createFileInitializer(filePath, fileType)

            if initializer is not None:
                item.setInitializer(initializer)

    def removeProbe(self, name: str) -> None:
        self._repository.removeItem(name)

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()


class ProbePresenter(Observable, Observer):

    def __init__(self, sizer: ProbeSizer, probe_: SelectedProbe, probeAPI: ProbeAPI) -> None:
        super().__init__()
        self._sizer = sizer
        self._probe = probe_
        self._probeAPI = probeAPI

    @classmethod
    def createInstance(cls, sizer: ProbeSizer, probe_: SelectedProbe,
                       probeAPI: ProbeAPI) -> ProbePresenter:
        presenter = cls(sizer, probe_, probeAPI)
        sizer.addObserver(presenter)
        probe_.addObserver(presenter)
        return presenter

    def isSelectedProbeValid(self) -> bool:
        selectedItem = self._probe.getSelectedItem()

        if selectedItem is None:
            return False

        probe = selectedItem.getProbe()
        actualExtent = probe.getExtentInPixels()
        expectedExtent = self._sizer.getExtentInPixels()
        return (actualExtent == expectedExtent)

    def selectProbe(self, name: str) -> None:
        self._probe.selectItem(name)

    def getSelectedProbe(self) -> str:
        return self._probe.getSelectedName()

    def getNumberOfProbeModes(self) -> int:
        selectedItem = self._probe.getSelectedItem()

        if selectedItem is None:
            return 0

        probe = selectedItem.getProbe()
        return probe.getNumberOfModes()

    def getSelectedProbeFlattenedArray(self) -> ProbeArrayType | None:
        selectedItem = self._probe.getSelectedItem()

        if selectedItem is None:
            return None

        probe = selectedItem.getProbe()
        return probe.getModesFlattened()

    def getSelectableNames(self) -> Sequence[str]:
        return self._probe.getSelectableNames()

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()


class ProbeCore(StatefulCore[ProbeStateData]):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 detector: Detector, diffractionPatternSizer: DiffractionPatternSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.settings = ProbeSettings.createInstance(settingsRegistry)
        self.sizer = ProbeSizer.createInstance(diffractionPatternSizer)
        self.apparatus = Apparatus.createInstance(detector, diffractionPatternSizer, self.settings)
        self.apparatusPresenter = ApparatusPresenter.createInstance(self.settings, self.apparatus)

        self._modesFactory = MultimodalProbeFactory(rng)
        self._factory = ProbeRepositoryItemFactory(self._modesFactory, self.settings,
                                                   self.apparatus, self.sizer, fileReaderChooser)
        self._repository = ProbeRepository()
        self._itemSettingsDelegate = ProbeRepositoryItemSettingsDelegate(
            self.settings, self._factory, self._repository)
        self._probe = SelectedProbe.createInstance(self._repository, self._itemSettingsDelegate,
                                                   settingsRegistry)
        self.probeAPI = ProbeAPI(self._factory, self._repository, self._probe)
        self.repositoryPresenter = ProbeRepositoryPresenter.createInstance(
            self._repository, self._factory, self.probeAPI, fileWriterChooser)
        self.presenter = ProbePresenter.createInstance(self.sizer, self._probe, self.probeAPI)

    def getStateData(self) -> ProbeStateData:
        pixelGeometry = self.apparatus.getObjectPlanePixelGeometry()
        probe = self.probeAPI.getSelectedProbe()
        return ProbeStateData(
            pixelSizeXInMeters=float(pixelGeometry.widthInMeters),
            pixelSizeYInMeters=float(pixelGeometry.heightInMeters),
            array=probe.getArray(),
        )

    def setStateData(self, stateData: ProbeStateData, stateFilePath: Path) -> None:
        probe = Probe(stateData.array)
        self.probeAPI.insertItemIntoRepository(name='Restart',
                                               probe=probe,
                                               filePath=stateFilePath,
                                               fileType=stateFilePath.suffix[1:],
                                               selectItem=True)
