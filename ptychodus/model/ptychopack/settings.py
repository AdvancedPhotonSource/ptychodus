from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPackSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPack')
        self._settings_group.addObserver(self)

        self.dm_exit_wave_relaxation = self._settings_group.createRealEntry(
            'dm_exit_wave_relaxation', '0.8')
        self.raar_exit_wave_relaxation = self._settings_group.createRealEntry(
            'raar_exit_wave_relaxation', '0.75')

        self.object_correction_plan_start = self._settings_group.createIntegerEntry(
            'object_correction_plan_start', 0)
        self.object_correction_plan_stop = self._settings_group.createIntegerEntry(
            'object_correction_plan_stop', 100)
        self.object_correction_plan_stride = self._settings_group.createIntegerEntry(
            'object_correction_plan_stride', 1)
        self.pie_alpha = self._settings_group.createRealEntry('pie_alpha', '1')
        self.pie_object_relaxation = self._settings_group.createRealEntry(
            'pie_object_relaxation', '1')

        self.probe_correction_plan_start = self._settings_group.createIntegerEntry(
            'probe_correction_plan_start', 10)
        self.probe_correction_plan_stop = self._settings_group.createIntegerEntry(
            'probe_correction_plan_stop', 100)
        self.probe_correction_plan_stride = self._settings_group.createIntegerEntry(
            'probe_correction_plan_stride', 1)
        self.pie_beta = self._settings_group.createRealEntry('pie_beta', '1')
        self.pie_probe_relaxation = self._settings_group.createRealEntry(
            'pie_probe_relaxation', '1')

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
