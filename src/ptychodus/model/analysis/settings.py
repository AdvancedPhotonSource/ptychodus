from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbePropagationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('ProbePropagation')
        self._settings_group.add_observer(self)

        self.begin_coordinate_m = self._settings_group.create_real_parameter(
            'BeginCoordinateInMeters', -1e-3
        )
        self.end_coordinate_m = self._settings_group.create_real_parameter(
            'EndCoordinateInMeters', 1e-3
        )
        self.num_steps = self._settings_group.create_integer_parameter('NumberOfSteps', 100)

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()
