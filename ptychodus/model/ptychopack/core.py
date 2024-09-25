from __future__ import annotations
from collections.abc import Iterator
from decimal import Decimal
import logging

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import NullReconstructor, Reconstructor, ReconstructorLibrary
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
        # FIXME device.addObserver(self)

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
        return f"Planned Iterations: {iterations}"

    def get_dm_exit_wave_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_dm_exit_wave_relaxation(self) -> Decimal:
        limits = self.get_dm_exit_wave_relaxation_limits()
        return limits.clamp(self._settings.dm_exit_wave_relaxation.getValue())

    def set_dm_exit_wave_relaxation(self, value: Decimal) -> None:
        self._settings.dm_exit_wave_relaxation.setValue(value)

    def get_raar_exit_wave_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_raar_exit_wave_relaxation(self) -> Decimal:
        limits = self.get_raar_exit_wave_relaxation_limits()
        return limits.clamp(self._settings.raar_exit_wave_relaxation.getValue())

    def set_raar_exit_wave_relaxation(self, value: Decimal) -> None:
        self._settings.raar_exit_wave_relaxation.setValue(value)

    def get_object_correction_plan_start(self) -> int:
        return self._settings.object_correction_plan_start.getValue()

    def set_object_correction_plan_start(self, value: int) -> None:
        self._settings.object_correction_plan_start.setValue(value)

    def get_object_correction_plan_stop(self) -> int:
        return self._settings.object_correction_plan_stop.getValue()

    def set_object_correction_plan_stop(self, value: int) -> None:
        self._settings.object_correction_plan_stop.setValue(value)

    def get_object_correction_plan_stride(self) -> int:
        return self._settings.object_correction_plan_stride.getValue()

    def set_object_correction_plan_stride(self, value: int) -> None:
        self._settings.object_correction_plan_stride.setValue(value)

    def get_pie_alpha_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_pie_alpha(self) -> Decimal:
        limits = self.get_pie_alpha_limits()
        return limits.clamp(self._settings.pie_alpha.getValue())

    def set_pie_alpha(self, value: Decimal) -> None:
        self._settings.pie_alpha.setValue(value)

    def get_pie_object_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_pie_object_relaxation(self) -> Decimal:
        limits = self.get_pie_object_relaxation_limits()
        return limits.clamp(self._settings.pie_object_relaxation.getValue())

    def set_pie_object_relaxation(self, value: Decimal) -> None:
        self._settings.pie_object_relaxation.setValue(value)

    def get_probe_correction_plan_start(self) -> int:
        return self._settings.probe_correction_plan_start.getValue()

    def set_probe_correction_plan_start(self, value: int) -> None:
        self._settings.probe_correction_plan_start.setValue(value)

    def get_probe_correction_plan_stop(self) -> int:
        return self._settings.probe_correction_plan_stop.getValue()

    def set_probe_correction_plan_stop(self, value: int) -> None:
        self._settings.probe_correction_plan_stop.setValue(value)

    def get_probe_correction_plan_stride(self) -> int:
        return self._settings.probe_correction_plan_stride.getValue()

    def set_probe_correction_plan_stride(self, value: int) -> None:
        self._settings.probe_correction_plan_stride.setValue(value)

    def get_pie_beta_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_pie_beta(self) -> Decimal:
        limits = self.get_pie_beta_limits()
        return limits.clamp(self._settings.pie_beta.getValue())

    def set_pie_beta(self, value: Decimal) -> None:
        self._settings.pie_beta.setValue(value)

    def get_pie_probe_relaxation_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_pie_probe_relaxation(self) -> Decimal:
        limits = self.get_pie_probe_relaxation_limits()
        return limits.clamp(self._settings.pie_probe_relaxation.getValue())

    def set_pie_probe_relaxation(self, value: Decimal) -> None:
        self._settings.pie_probe_relaxation.setValue(value)

    def get_position_correction_plan_start(self) -> int:
        return self._settings.position_correction_plan_start.getValue()

    def set_position_correction_plan_start(self, value: int) -> None:
        self._settings.position_correction_plan_start.setValue(value)

    def get_position_correction_plan_stop(self) -> int:
        return self._settings.position_correction_plan_stop.getValue()

    def set_position_correction_plan_stop(self, value: int) -> None:
        self._settings.position_correction_plan_stop.setValue(value)

    def get_position_correction_plan_stride(self) -> int:
        return self._settings.position_correction_plan_stride.getValue()

    def set_position_correction_plan_stride(self, value: int) -> None:
        self._settings.position_correction_plan_stride.setValue(value)

    def get_position_correction_probe_threshold_limits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def get_position_correction_probe_threshold(self) -> Decimal:
        limits = self.get_position_correction_probe_threshold_limits()
        return limits.clamp(self._settings.position_correction_probe_threshold.getValue())

    def set_position_correction_probe_threshold(self, value: Decimal) -> None:
        self._settings.position_correction_probe_threshold.setValue(value)

    def get_position_correction_feedback(self) -> Decimal:
        return self._settings.position_correction_feedback.getValue()

    def set_position_correction_feedback(self, value: Decimal) -> None:
        self._settings.position_correction_feedback.setValue(value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        # FIXME elif observable is self._device:
        # FIXME     self.notifyObservers()


class PtychoPackReconstructorLibrary(ReconstructorLibrary):
    def __init__(self, settingsRegistry: SettingsRegistry, isDeveloperModeEnabled: bool) -> None:
        super().__init__()
        self._settings = PtychoPackSettings(settingsRegistry)
        self._device: PtychoPackDevice = NullPtychoPackDevice()
        self.reconstructor_list: list[Reconstructor] = list()

        try:
            from .dm import DifferenceMapReconstructor
            from .pie import PtychographicIterativeEngineReconstructor
            from .raar import RelaxedAveragedAlternatingReflectionsReconstructor
            from .real_device import RealPtychoPackDevice
        except ModuleNotFoundError:
            logger.info("PtychoPack not found.")

            if isDeveloperModeEnabled:
                self.reconstructor_list.append(NullReconstructor("PIE"))
                self.reconstructor_list.append(NullReconstructor("DM"))
                self.reconstructor_list.append(NullReconstructor("RAAR"))
        else:
            self._device = RealPtychoPackDevice()
            self.reconstructor_list.append(
                PtychographicIterativeEngineReconstructor(self._settings, self._device)
            )
            self.reconstructor_list.append(
                DifferenceMapReconstructor(self._settings, self._device)
            )
            self.reconstructor_list.append(
                RelaxedAveragedAlternatingReflectionsReconstructor(self._settings, self._device)
            )

        self.presenter = PtychoPackPresenter(self._settings, self._device)

    @property
    def name(self) -> str:
        return "PtychoPack"

    def __iter__(self) -> Iterator[Reconstructor]:
        return iter(self.reconstructor_list)
