from __future__ import annotations
from collections.abc import Sequence
import logging

from ...api.observer import Observable, Observer
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructorLibrary
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

    def __init__(self, activeReconstructor: ActiveReconstructor,
                 reconstructorAPI: ReconstructorAPI) -> None:
        super().__init__()
        self._activeReconstructor = activeReconstructor
        self._reconstructorAPI = reconstructorAPI
        self._plot2D = Plot2D.createNull()

    @classmethod
    def createInstance(cls, activeReconstructor: ActiveReconstructor,
                       reconstructorAPI: ReconstructorAPI) -> ReconstructorPresenter:
        presenter = cls(activeReconstructor, reconstructorAPI)
        activeReconstructor.addObserver(presenter)
        return presenter

    def getReconstructorList(self) -> Sequence[str]:
        return self._activeReconstructor.getReconstructorList()

    def getReconstructor(self) -> str:
        return self._activeReconstructor.name

    def setReconstructor(self, name: str) -> None:
        self._activeReconstructor.selectReconstructor(name)

    def reconstruct(self) -> None:
        result = self._reconstructorAPI.reconstruct()
        self._plot2D = result.plot2D
        self.notifyObservers()

    def reconstructSplit(self) -> None:
        seriesXList: list[PlotSeries] = list()
        seriesYList: list[PlotSeries] = list()

        resultOdd, resultEven = self._reconstructorAPI.reconstructSplit()

        for evenOdd, plot2D in zip(('Odd', 'Even'), (resultOdd.plot2D, resultEven.plot2D)):
            for seriesX in plot2D.axisX.series:
                for seriesY in plot2D.axisY.series:
                    seriesXList.append(PlotSeries(f'{seriesX.label} - {evenOdd}', seriesX.values))
                    seriesYList.append(PlotSeries(f'{seriesY.label} - {evenOdd}', seriesY.values))

        self._plot2D = Plot2D(
            axisX=PlotAxis(label=plot2D.axisX.label, series=seriesXList),
            axisY=PlotAxis(label=plot2D.axisY.label, series=seriesYList),
        )
        self.notifyObservers()

    def getPlot(self) -> Plot2D:
        return self._plot2D

    @property
    def isTrainable(self) -> bool:
        return self._activeReconstructor.isTrainable

    def ingest(self) -> None:
        self._reconstructorAPI.ingest()

    def train(self) -> None:
        self._plot2D = self._reconstructorAPI.train()
        self.notifyObservers()

    def reset(self) -> None:
        self._reconstructorAPI.reset()

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
        self.presenter = ReconstructorPresenter.createInstance(self._activeReconstructor,
                                                               self.reconstructorAPI)
