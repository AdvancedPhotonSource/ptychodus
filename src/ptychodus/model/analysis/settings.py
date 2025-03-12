from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbePropagationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('ProbePropagation')
        self._settingsGroup.add_observer(self)

        self.beginCoordinateInMeters = self._settingsGroup.create_real_parameter(
            'BeginCoordinateInMeters', -1e-3
        )
        self.endCoordinateInMeters = self._settingsGroup.create_real_parameter(
            'EndCoordinateInMeters', 1e-3
        )
        self.numberOfSteps = self._settingsGroup.create_integer_parameter('NumberOfSteps', 100)

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
