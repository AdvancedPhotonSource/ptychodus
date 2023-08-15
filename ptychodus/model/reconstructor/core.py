from __future__ import annotations
from collections.abc import Sequence
import logging

from ...api.observer import Observable, Observer
from ...api.reconstructor import ReconstructOutput, ReconstructorLibrary
from ...api.settings import SettingsRegistry
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..scan import ScanAPI
from .active import ActiveReconstructor
from .api import ReconstructorAPI
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):

    def __init__(self, activeReconstructor: ActiveReconstructor) -> None:
        super().__init__()
        self._activeReconstructor = activeReconstructor

    @classmethod
    def createInstance(cls, activeReconstructor: ActiveReconstructor) -> ReconstructorPresenter:
        presenter = cls(activeReconstructor)
        activeReconstructor.addObserver(presenter)
        return presenter

    def getReconstructorList(self) -> Sequence[str]:
        return self._activeReconstructor.getReconstructorList()

    def getReconstructor(self) -> str:
        return self._activeReconstructor.name

    def setReconstructor(self, name: str) -> None:
        self._activeReconstructor.selectReconstructor(name)

    def reconstruct(self) -> ReconstructOutput:
        label = self._activeReconstructor.name
        return self._activeReconstructor.reconstruct(label)

    @property
    def isTrainable(self) -> bool:
        return self._activeReconstructor.isTrainable

    def ingest(self) -> None:
        self._activeReconstructor.ingest()

    def train(self) -> None:
        self._activeReconstructor.train()

    def reset(self) -> None:
        self._activeReconstructor.reset()

    def update(self, observable: Observable) -> None:
        if observable is self._activeReconstructor:
            self.notifyObservers()


class ReconstructorCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                 libraryList: Sequence[ReconstructorLibrary]) -> None:
        self.settings = ReconstructorSettings.createInstance(settingsRegistry)
        self._activeReconstructor = ActiveReconstructor.createInstance(
            self.settings, diffractionDataset, scanAPI, probeAPI, objectAPI, libraryList,
            settingsRegistry)
        self.reconstructorAPI = ReconstructorAPI(self._activeReconstructor)
        self.presenter = ReconstructorPresenter.createInstance(self._activeReconstructor)
