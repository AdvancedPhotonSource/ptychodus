from __future__ import annotations
from collections.abc import Iterator, Sequence
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from .settings import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikePositionCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)

logger = logging.getLogger(__name__)


class TikePresenter(Observable, Observer):

    def __init__(self, settings: TikeSettings) -> None:
        super().__init__()
        self._settings = settings
        self._logger = logging.getLogger("tike")

        settings.addObserver(self)

    def getNumGpus(self) -> str:
        return self._settings.numGpus.getValue()

    def setNumGpus(self, value: str) -> None:
        self._settings.numGpus.setValue(value)

    def getNoiseModelList(self) -> Sequence[str]:
        return ["poisson", "gaussian"]

    def getNoiseModel(self) -> str:
        return self._settings.noiseModel.getValue()

    def setNoiseModel(self, name: str) -> None:
        self._settings.noiseModel.setValue(name)

    def getBatchMethodList(self) -> Sequence[str]:
        return ["wobbly_center", "wobbly_center_random_bootstrap", "compact"]

    def getBatchMethod(self) -> str:
        return self._settings.batchMethod.getValue()

    def setBatchMethod(self, name: str) -> None:
        self._settings.batchMethod.setValue(name)

    def getLogLevelList(self) -> Sequence[str]:
        return ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

    def getLogLevel(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def setLogLevel(self, name: str) -> None:
        nameBefore = self.getLogLevel()

        if name == nameBefore:
            return

        try:
            self._logger.setLevel(name)
        except ValueError:
            logger.error(f'Bad log level "{name}".')

        nameAfter = self.getLogLevel()
        logger.info(f"Changed Tike log level {nameBefore} -> {nameAfter}")
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        super().__init__()
        self._settings = TikeSettings(settingsRegistry)
        self._multigridSettings = TikeMultigridSettings(settingsRegistry)
        self._positionCorrectionSettings = TikePositionCorrectionSettings(settingsRegistry)
        self._probeCorrectionSettings = TikeProbeCorrectionSettings(settingsRegistry)
        self._objectCorrectionSettings = TikeObjectCorrectionSettings(settingsRegistry)
        self.presenter = TikePresenter(self._settings)

        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> TikeReconstructorLibrary:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import IterativeLeastSquaresReconstructor
            from .reconstructor import RegularizedPIEReconstructor
            from .reconstructor import TikeReconstructor
        except ModuleNotFoundError:
            logger.info("Tike not found.")

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor("rpie"))
                core.reconstructorList.append(NullReconstructor("lstsq_grad"))
        else:
            tikeReconstructor = TikeReconstructor(
                core._settings,
                core._multigridSettings,
                core._positionCorrectionSettings,
                core._probeCorrectionSettings,
                core._objectCorrectionSettings,
            )
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))

        return core

    @property
    def name(self) -> str:
        return "Tike"

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
