from __future__ import annotations
from collections.abc import Iterator
from decimal import Decimal
import logging

from ptychodus.api.geometry import Interval
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

    def get_device(self) -> str:
        return ''  # FIXME

    def set_device(self, device: str) -> None:
        pass  # FIXME

    def get_plan(self) -> str:
        return 'Planned Iterations: 999'  # FIXME

    def get_object_correction_plan_start(self) -> int:
        return self._settings.object_correction_plan_start.value

    def set_object_correction_plan_start(self, value: int) -> None:
        self._settings.object_correction_plan_start.value = value

    def get_object_correction_plan_stop(self) -> int:
        return self._settings.object_correction_plan_stop.value

    def set_object_correction_plan_stop(self, value: int) -> None:
        self._settings.object_correction_plan_stop.value = value

    def get_object_correction_plan_stride(self) -> int:
        return self._settings.object_correction_plan_stride.value

    def set_object_correction_plan_stride(self, value: int) -> None:
        self._settings.object_correction_plan_stride.value = value

    def get_alpha_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_alpha(self) -> Decimal:
        limits = self.get_alpha_limits()
        return limits.clamp(self._settings.alpha.value)

    def set_alpha(self, value: Decimal) -> None:
        self._settings.alpha.value = value

    def get_object_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_object_relaxation(self) -> Decimal:
        limits = self.get_object_relaxation_limits()
        return limits.clamp(self._settings.object_relaxation.value)

    def set_object_relaxation(self, value: Decimal) -> None:
        self._settings.object_relaxation.value = value

    def get_probe_correction_plan_start(self) -> int:
        return self._settings.probe_correction_plan_start.value

    def set_probe_correction_plan_start(self, value: int) -> None:
        self._settings.probe_correction_plan_start.value = value

    def get_probe_correction_plan_stop(self) -> int:
        return self._settings.probe_correction_plan_stop.value

    def set_probe_correction_plan_stop(self, value: int) -> None:
        self._settings.probe_correction_plan_stop.value = value

    def get_probe_correction_plan_stride(self) -> int:
        return self._settings.probe_correction_plan_stride.value

    def set_probe_correction_plan_stride(self, value: int) -> None:
        self._settings.probe_correction_plan_stride.value = value

    def get_probe_power_correction_plan_start(self) -> int:
        return self._settings.probe_power_correction_plan_start.value

    def set_probe_power_correction_plan_start(self, value: int) -> None:
        self._settings.probe_power_correction_plan_start.value = value

    def get_probe_power_correction_plan_stop(self) -> int:
        return self._settings.probe_power_correction_plan_stop.value

    def set_probe_power_correction_plan_stop(self, value: int) -> None:
        self._settings.probe_power_correction_plan_stop.value = value

    def get_probe_power_correction_plan_stride(self) -> int:
        return self._settings.probe_power_correction_plan_stride.value

    def set_probe_power_correction_plan_stride(self, value: int) -> None:
        self._settings.probe_power_correction_plan_stride.value = value

    def get_beta_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_beta(self) -> Decimal:
        limits = self.get_beta_limits()
        return limits.clamp(self._settings.beta.value)

    def set_beta(self, value: Decimal) -> None:
        self._settings.beta.value = value

    def get_probe_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_probe_relaxation(self) -> Decimal:
        limits = self.get_probe_relaxation_limits()
        return limits.clamp(self._settings.probe_relaxation.value)

    def set_probe_relaxation(self, value: Decimal) -> None:
        self._settings.probe_relaxation.value = value

    def get_position_correction_plan_start(self) -> int:
        return self._settings.position_correction_plan_start.value

    def set_position_correction_plan_start(self, value: int) -> None:
        self._settings.position_correction_plan_start.value = value

    def get_position_correction_plan_stop(self) -> int:
        return self._settings.position_correction_plan_stop.value

    def set_position_correction_plan_stop(self, value: int) -> None:
        self._settings.position_correction_plan_stop.value = value

    def get_position_correction_plan_stride(self) -> int:
        return self._settings.position_correction_plan_stride.value

    def set_position_correction_plan_stride(self, value: int) -> None:
        self._settings.position_correction_plan_stride.value = value

    def get_position_correction_probe_threshold_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_position_correction_probe_threshold(self) -> Decimal:
        limits = self.get_position_correction_probe_threshold_limits()
        return limits.clamp(self._settings.position_correction_probe_threshold.value)

    def set_position_correction_probe_threshold(self, value: Decimal) -> None:
        self._settings.position_correction_probe_threshold.value = value

    def get_position_correction_feedback(self) -> Decimal:
        return self._settings.position_correction_feedback.value

    def set_position_correction_feedback(self, value: Decimal) -> None:
        self._settings.position_correction_feedback.value = value

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
