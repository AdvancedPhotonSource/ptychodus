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

    def get_available_devices(self) -> Iterator[str]:
        return iter([])  # FIXME

    def set_device(self, device: str) -> None:
        pass  # FIXME

    def get_device(self) -> str:
        return ''  # FIXME

    def get_plan(self) -> str:
        return 'Planned Iterations: 999'  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class PtychoPackReconstructorLibrary(ReconstructorLibrary):

    def __init__(self, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool) -> None:
        super().__init__()
        self._settings = PtychoPackSettings(settingsRegistry)
        self.presenter = PtychoPackPresenter(self._settings)
        self.reconstructor_list: list[Reconstructor] = list()

        try:
            from .pie import PtychographicIterativeEngineReconstructor
        except ModuleNotFoundError:
            logger.info('PtychoPack not found.')

            if isDeveloperModeEnabled:
                self.reconstructor_list.append(NullReconstructor('PIE'))
        else:
            self.reconstructor_list.append(PtychographicIterativeEngineReconstructor())

    @property
    def name(self) -> str:
        return 'PtychoPack'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
