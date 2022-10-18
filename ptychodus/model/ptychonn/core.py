from __future__ import annotations
from pathlib import Path
from typing import Final, Iterator
import logging

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import ActiveDiffractionDataset
from ..object import Object
from ..probe import Apparatus
from ..scan import Scan
from .settings import PtychoNNSettings

logger = logging.getLogger(__name__)


class PtychoNNPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PtychoNNSettings) -> None:
        super().__init__()
        self._settings = settings
        self._fileFilterList: list[str] = ['PyTorch Model State Files (*.pt *.pth)']

    @classmethod
    def createInstance(cls, settings: PtychoNNSettings) -> PtychoNNPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getFileFilterList(self) -> list[str]:
        return self._fileFilterList

    def getFileFilter(self) -> str:
        return self._fileFilterList[0]

    def getModelStateFilePath(self) -> Path:
        return self._settings.modelStateFilePath.value

    def setModelStateFilePath(self, directory: Path) -> None:
        self._settings.modelStateFilePath.value = directory

    def getBatchSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getBatchSize(self) -> int:
        limits = self.getBatchSizeLimits()
        return limits.clamp(self._settings.batchSize.value)

    def setBatchSize(self, value: int) -> None:
        self._settings.batchSize.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoNNReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        super().__init__()
        self._settings = PtychoNNSettings.createInstance(settingsRegistry)
        self.presenter = PtychoNNPresenter.createInstance(self._settings)
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls,
                       settingsRegistry: SettingsRegistry,
                       diffractionDataset: ActiveDiffractionDataset,
                       scan: Scan,
                       apparatus: Apparatus,
                       object_: Object,
                       isDeveloperModeEnabled: bool = False) -> PtychoNNReconstructorLibrary:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import PtychoNNReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoNN not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('PtychoNN'))
        else:
            core.reconstructorList.append(
                PtychoNNReconstructor(core._settings, apparatus, scan, object_,
                                      diffractionDataset))

        return core

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
