from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPackSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoPack')
        self._settingsGroup.addObserver(self)

        self.dm_exit_wave_relaxation = self._settingsGroup.createRealParameter(
            'dm_exit_wave_relaxation', 0.8, minimum=0.0, maximum=1.0
        )
        self.raar_exit_wave_relaxation = self._settingsGroup.createRealParameter(
            'raar_exit_wave_relaxation', 0.75, minimum=0.0, maximum=1.0
        )

        self.object_correction_plan_start = self._settingsGroup.createIntegerParameter(
            'object_correction_plan_start', 0, minimum=0
        )
        self.object_correction_plan_stop = self._settingsGroup.createIntegerParameter(
            'object_correction_plan_stop', 100, minimum=0
        )
        self.object_correction_plan_stride = self._settingsGroup.createIntegerParameter(
            'object_correction_plan_stride', 1, minimum=1
        )
        self.pie_alpha = self._settingsGroup.createRealParameter(
            'pie_alpha', 1.0, minimum=0.0, maximum=1.0
        )
        self.pie_object_relaxation = self._settingsGroup.createRealParameter(
            'pie_object_relaxation', 1.0, minimum=0.0, maximum=1.0
        )

        self.probe_correction_plan_start = self._settingsGroup.createIntegerParameter(
            'probe_correction_plan_start', 10, minimum=0
        )
        self.probe_correction_plan_stop = self._settingsGroup.createIntegerParameter(
            'probe_correction_plan_stop', 100, minimum=0
        )
        self.probe_correction_plan_stride = self._settingsGroup.createIntegerParameter(
            'probe_correction_plan_stride', 1, minimum=1
        )
        self.pie_beta = self._settingsGroup.createRealParameter(
            'pie_beta', 1.0, minimum=0.0, maximum=1.0
        )
        self.pie_probe_relaxation = self._settingsGroup.createRealParameter(
            'pie_probe_relaxation', 1.0, minimum=0.0, maximum=1.0
        )

        self.position_correction_plan_start = self._settingsGroup.createIntegerParameter(
            'position_correction_plan_start', 0, minimum=0
        )
        self.position_correction_plan_stop = self._settingsGroup.createIntegerParameter(
            'position_correction_plan_stop', 0, minimum=0
        )
        self.position_correction_plan_stride = self._settingsGroup.createIntegerParameter(
            'position_correction_plan_stride', 1, minimum=1
        )
        self.position_correction_probe_threshold = self._settingsGroup.createRealParameter(
            'position_correction_probe_threshold',
            0.1,
            minimum=0.0,
            maximum=1.0,
        )
        self.position_correction_feedback = self._settingsGroup.createRealParameter(
            'position_correction_feedback', 50.0, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
