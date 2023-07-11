from __future__ import annotations
from collections.abc import Sequence
import logging

from ...api.observer import Observable, Observer
from ...api.reconstructor import ReconstructOutput, ReconstructorLibrary, TrainableReconstructor
from ...api.settings import SettingsRegistry
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..scan import ScanAPI
from .reconstructor import ReconstructorRepository, ActiveReconstructor
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings, repository: ReconstructorRepository,
                 activeReconstructor: ActiveReconstructor) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._activeReconstructor = activeReconstructor

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings, repository: ReconstructorRepository,
                       activeReconstructor: ActiveReconstructor) -> ReconstructorPresenter:
        presenter = cls(settings, repository, activeReconstructor)
        repository.addObserver(presenter)
        activeReconstructor.addObserver(presenter)
        return presenter

    def getReconstructorList(self) -> Sequence[str]:
        return list(self._repository.keys())

    def getReconstructor(self) -> str:
        return self._activeReconstructor.name

    def setReconstructor(self, name: str) -> None:
        self._activeReconstructor.selectReconstructor(name)

    @property
    def isTrainable(self) -> bool:
        return self._activeReconstructor.isTrainable

    def train(self) -> None:
        recon = self._activeReconstructor

        # FIXME as implemented, this is always false
        if isinstance(recon, TrainableReconstructor):
            recon.train()
        else:
            logger.error('Reconstructor is not trainable!')

    def execute(self) -> ReconstructOutput:
        return self._activeReconstructor.execute('Result')  # FIXME better name

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()
        elif observable is self._activeReconstructor:
            self.notifyObservers()


class ReconstructorPlotPresenter(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._xlabel: str = 'Iteration'
        self._xvalues: Sequence[Sequence[float]] = list()
        self._ylabel: str = 'Objective'
        self._yvalues: Sequence[Sequence[float]] = list()

    @property
    def xlabel(self) -> str:
        return self._xlabel

    @xlabel.setter
    def xlabel(self, value: str) -> None:
        if self._xlabel != value:
            self._xlabel = value
            self.notifyObservers()

    @property
    def ylabel(self) -> str:
        return self._ylabel

    @ylabel.setter
    def ylabel(self, value: str) -> None:
        if self._ylabel != value:
            self._ylabel = value
            self.notifyObservers()

    @property
    def xvalues(self) -> Sequence[Sequence[float]]:
        return self._xvalues

    @property
    def yvalues(self) -> Sequence[Sequence[float]]:
        return self._yvalues

    def setValues(self, xvalues: Sequence[Sequence[float]],
                  yvalues: Sequence[Sequence[float]]) -> None:
        self._xvalues = xvalues
        self._yvalues = yvalues
        self.notifyObservers()

    def setEnumeratedYValues(self, yvalues: Sequence[Sequence[float]]) -> None:
        xvalues = [*range(len(yvalues))]
        self.setValues(xvalues, yvalues)


class ReconstructorCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                 libraryList: list[ReconstructorLibrary]) -> None:
        self.settings = ReconstructorSettings.createInstance(settingsRegistry)
        self._repository = ReconstructorRepository.createInstance(libraryList)
        self._activeReconstructor = ActiveReconstructor.createInstance(
            self.settings, self._repository, diffractionDataset, scanAPI, probeAPI, objectAPI,
            settingsRegistry)
        self.presenter = ReconstructorPresenter.createInstance(self.settings, self._repository,
                                                               self._activeReconstructor)
        self.plotPresenter = ReconstructorPlotPresenter()
