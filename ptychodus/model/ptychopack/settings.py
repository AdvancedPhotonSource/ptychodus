from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPackSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPack')
        self._settings_group.addObserver(self)

        self.object_correction_plan_start = self._settings_group.createIntegerEntry(
            'object_correction_plan_start', 0)
        self.object_correction_plan_stop = self._settings_group.createIntegerEntry(
            'object_correction_plan_stop', 100)
        self.object_correction_plan_stride = self._settings_group.createIntegerEntry(
            'object_correction_plan_stride', 1)
        self.alpha = self._settings_group.createRealEntry('alpha', '1')
        self.object_relaxation = self._settings_group.createRealEntry('object_relaxation', '1')

        self.probe_correction_plan_start = self._settings_group.createIntegerEntry(
            'probe_correction_plan_start', 10)
        self.probe_correction_plan_stop = self._settings_group.createIntegerEntry(
            'probe_correction_plan_stop', 100)
        self.probe_correction_plan_stride = self._settings_group.createIntegerEntry(
            'probe_correction_plan_stride', 1)
        self.probe_power_correction_plan_start = self._settings_group.createIntegerEntry(
            'probe_power_correction_plan_start', 0)
        self.probe_power_correction_plan_stop = self._settings_group.createIntegerEntry(
            'probe_power_correction_plan_stop', 1)
        self.probe_power_correction_plan_stride = self._settings_group.createIntegerEntry(
            'probe_power_correction_plan_stride', 10)
        self.beta = self._settings_group.createRealEntry('beta', '1')
        self.probe_relaxation = self._settings_group.createRealEntry('probe_relaxation', '1')

        self.position_correction_plan_start = self._settings_group.createIntegerEntry(
            'position_correction_plan_start', 0)
        self.position_correction_plan_stop = self._settings_group.createIntegerEntry(
            'position_correction_plan_stop', 0)
        self.position_correction_plan_stride = self._settings_group.createIntegerEntry(
            'position_correction_plan_stride', 1)
        self.position_correction_probe_threshold = self._settings_group.createRealEntry(
            'position_correction_probe_threshold', '0.1')
        self.position_correction_feedback = self._settings_group.createRealEntry(
            'position_correction_feedback', '50')

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()
