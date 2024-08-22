from __future__ import annotations
from collections.abc import Iterator
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
from ptychodus.api.settings import SettingsRegistry

from .settings import PtychoPackSettings

logger = logging.getLogger(__name__)


class PtychoPackPresenter(Observable, Observer):

    def __init__(self, settings: PtychoPackSettings) -> None:
        super().__init__()
        self._settings = settings
        settings.addObserver(self)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPackReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        super().__init__()
        self._settings = PtychoPackSettings(settingsRegistry)
        self.presenter = PtychoPackPresenter(self._settings)
        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       isDeveloperModeEnabled: bool) -> PtychoPackReconstructorLibrary:
        core = cls(settingsRegistry)

        try:
            from .pie import PtychographicIterativeEngineReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPack not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rPIE'))
        else:
            core.reconstructorList.append(PtychographicIterativeEngineReconstructor())

        return core

    @property
    def name(self) -> str:
        return 'PtychoPack'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructorList)
