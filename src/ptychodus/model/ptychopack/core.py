from __future__ import annotations
from collections.abc import Iterator
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import (
    NullReconstructor,
    Reconstructor,
    ReconstructorLibrary,
)
from ptychodus.api.settings import SettingsRegistry

from .device import NullPtychoPackDevice, PtychoPackDevice
from .settings import PtychoPackSettings

logger = logging.getLogger(__name__)


class PtychoPackPresenter(Observable, Observer):
    def __init__(self, settings: PtychoPackSettings, device: PtychoPackDevice) -> None:
        super().__init__()
        self._settings = settings
        self._device = device
        settings.addObserver(self)
        device.addObserver(self)

    def get_available_devices(self) -> Iterator[str]:
        return self._device.get_available_devices()

    def get_device(self) -> str:
        return self._device.get_device()

    def set_device(self, device: str) -> None:
        self._device.set_device(device)

    def get_plan(self) -> str:
        iterations = self._settings.object_correction_plan_stop.getValue()
        iterations = max(iterations, self._settings.probe_correction_plan_stop.getValue())
        iterations = max(iterations, self._settings.position_correction_plan_stop.getValue())
        return f'Planned Iterations: {iterations}'

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._device:
            self.notifyObservers()


class PtychoPackReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool) -> None:
        super().__init__()
        self.settings = PtychoPackSettings(settingsRegistry)
        self._device: PtychoPackDevice = NullPtychoPackDevice()
        self.reconstructor_list: list[Reconstructor] = list()

        try:
            from .dm import DifferenceMapReconstructor
            from .pie import PtychographicIterativeEngineReconstructor
            from .raar import RelaxedAveragedAlternatingReflectionsReconstructor
            from .real_device import RealPtychoPackDevice
        except ModuleNotFoundError:
            logger.info('PtychoPack not found.')

            if isDeveloperModeEnabled:
                self.reconstructor_list.append(NullReconstructor('PIE'))
                self.reconstructor_list.append(NullReconstructor('DM'))
                self.reconstructor_list.append(NullReconstructor('RAAR'))
        else:
            self._device = RealPtychoPackDevice()
            self.reconstructor_list.append(
                PtychographicIterativeEngineReconstructor(self.settings, self._device)
            )
            self.reconstructor_list.append(DifferenceMapReconstructor(self.settings, self._device))
            self.reconstructor_list.append(
                RelaxedAveragedAlternatingReflectionsReconstructor(self.settings, self._device)
            )

        self.presenter = PtychoPackPresenter(self.settings, self._device)

    @property
    def name(self) -> str:
        return 'PtychoPack'

    @property
    def logger_name(self) -> str:
        return 'ptychopack'

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
