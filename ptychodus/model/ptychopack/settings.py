from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter, RealParameter
from ptychodus.api.settings import SettingsRegistry


class PtychoPackSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPack')
        self._settings_group.addObserver(self)

        self.dm_exit_wave_relaxation = RealParameter(
            self._settings_group, 'dm_exit_wave_relaxation', 0.8, minimum=0.0, maximum=1.0
        )
        self.raar_exit_wave_relaxation = RealParameter(
            self._settings_group, 'raar_exit_wave_relaxation', 0.75, minimum=0.0, maximum=1.0
        )

        self.object_correction_plan_start = IntegerParameter(
            self._settings_group, 'object_correction_plan_start', 0, minimum=0
        )
        self.object_correction_plan_stop = IntegerParameter(
            self._settings_group, 'object_correction_plan_stop', 100, minimum=0
        )
        self.object_correction_plan_stride = IntegerParameter(
            self._settings_group, 'object_correction_plan_stride', 1, minimum=1
        )
        self.pie_alpha = RealParameter(
            self._settings_group, 'pie_alpha', 1.0, minimum=0.0, maximum=1.0
        )
        self.pie_object_relaxation = RealParameter(
            self._settings_group, 'pie_object_relaxation', 1.0, minimum=0.0, maximum=1.0
        )

        self.probe_correction_plan_start = IntegerParameter(
            self._settings_group, 'probe_correction_plan_start', 10, minimum=0
        )
        self.probe_correction_plan_stop = IntegerParameter(
            self._settings_group, 'probe_correction_plan_stop', 100, minimum=0
        )
        self.probe_correction_plan_stride = IntegerParameter(
            self._settings_group, 'probe_correction_plan_stride', 1, minimum=1
        )
        self.pie_beta = RealParameter(
            self._settings_group, 'pie_beta', 1.0, minimum=0.0, maximum=1.0
        )
        self.pie_probe_relaxation = RealParameter(
            self._settings_group, 'pie_probe_relaxation', 1.0, minimum=0.0, maximum=1.0
        )

        self.position_correction_plan_start = IntegerParameter(
            self._settings_group, 'position_correction_plan_start', 0, minimum=0
        )
        self.position_correction_plan_stop = IntegerParameter(
            self._settings_group, 'position_correction_plan_stop', 0, minimum=0
        )
        self.position_correction_plan_stride = IntegerParameter(
            self._settings_group, 'position_correction_plan_stride', 1, minimum=1
        )
        self.position_correction_probe_threshold = RealParameter(
            self._settings_group,
            'position_correction_probe_threshold',
            0.1,
            minimum=0.0,
            maximum=1.0,
        )
        self.position_correction_feedback = RealParameter(
            self._settings_group, 'position_correction_feedback', 50.0, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()
