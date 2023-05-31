from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import ProbeArrayType, ProbeFileReader, ProbeFileWriter
from ...api.settings import SettingsRegistry
from ..data import DiffractionPatternSizer
from ..detector import Detector
from ..statefulCore import StateDataType, StatefulCore
from .api import ProbeAPI
from .apparatus import Apparatus, ApparatusPresenter
from .factory import ProbeRepositoryItemFactory
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

    def initializeProbe(self, displayName: str) -> Optional[str]:
        return self._probeAPI.insertItemIntoRepositoryFromInitializerDisplayName(displayName)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._probeAPI.insertItemIntoRepositoryFromFile(filePath, displayFileType=fileFilter)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveProbe(self, name: str, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, item.getArray())

        # TODO test this
        if item.getInitializer() is None:
            initializer = self._itemFactory.createFileInitializer(filePath,
                                                                  simpleFileType=fileType)

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
        selectedProbe = self._probe.getSelectedItem()

        if selectedProbe is None:
            return False

        actualExtent = selectedProbe.getExtentInPixels()
        expectedExtent = self._sizer.getProbeExtent()
        hasComplexDataType = numpy.iscomplexobj(selectedProbe.getArray())
        return (actualExtent == expectedExtent and hasComplexDataType)

    def selectProbe(self, name: str) -> None:
        self._probeAPI.selectItem(name)

    def getSelectedProbe(self) -> str:
        return self._probe.getSelectedName()

    def getNumberOfProbeModes(self) -> int:
        selectedProbe = self._probe.getSelectedItem()
        return 0 if selectedProbe is None else selectedProbe.getNumberOfModes()

    def getSelectedProbeFlattenedArray(self) -> Optional[ProbeArrayType]:
        selectedProbe = self._probe.getSelectedItem()
        return None if selectedProbe is None else selectedProbe.getModesFlattened()

    def getSelectableNames(self) -> Sequence[str]:
        return self._probe.getSelectableNames()

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()


class ProbeCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 detector: Detector, diffractionPatternSizer: DiffractionPatternSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        self.settings = ProbeSettings.createInstance(settingsRegistry)
        self.sizer = ProbeSizer.createInstance(diffractionPatternSizer)
        self.apparatus = Apparatus.createInstance(detector, diffractionPatternSizer, self.settings)
        self.apparatusPresenter = ApparatusPresenter.createInstance(self.settings, self.apparatus)

        self._factory = ProbeRepositoryItemFactory(rng, self.settings, self.apparatus, self.sizer,
                                                   fileReaderChooser)
        self._repository = ProbeRepository()
        self._itemSettingsDelegate = ProbeRepositoryItemSettingsDelegate(
            self.settings, self._factory, self._repository)
        self._probe = SelectedProbe.createInstance(self._repository, self._itemSettingsDelegate,
                                                   settingsRegistry)
        self.probeAPI = ProbeAPI(self._factory, self._repository, self._probe)
        self.repositoryPresenter = ProbeRepositoryPresenter.createInstance(
            self._repository, self._factory, self.probeAPI, fileWriterChooser)
        self.presenter = ProbePresenter.createInstance(self.sizer, self._probe, self.probeAPI)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        pixelSizeXInMeters = float(self.apparatus.getObjectPlanePixelSizeXInMeters())
        pixelSizeYInMeters = float(self.apparatus.getObjectPlanePixelSizeYInMeters())
        selectedProbeArray = self.probeAPI.getSelectedProbeArray()

        if selectedProbeArray is not None:
            state: StateDataType = {
                'pixelSizeXInMeters': numpy.array([pixelSizeXInMeters]),
                'pixelSizeYInMeters': numpy.array([pixelSizeYInMeters]),
                'probe': selectedProbeArray,
            }
        return state

    def setStateData(self, state: StateDataType) -> None:
        try:
            array = state['probe']
        except KeyError:
            logger.debug('Failed to restore probe array state!')
            return

        filePath = Path(''.join(state['restartFilePath']))
        itemName = self.probeAPI.insertItemIntoRepositoryFromArray(nameHint='Restart',
                                                                   array=array,
                                                                   filePath=filePath,
                                                                   simpleFileType='NPZ')

        if itemName is None:
            logger.error('Failed to initialize \"{name}\"!')
        else:
            self.probeAPI.selectItem(itemName)
