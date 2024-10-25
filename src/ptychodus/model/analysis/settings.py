from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbePropagationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('ProbePropagation')
        self._settingsGroup.addObserver(self)

        self.beginCoordinateInMeters = self._settingsGroup.createRealParameter(
            'BeginCoordinateInMeters', -1e-3
        )
        self.endCoordinateInMeters = self._settingsGroup.createRealParameter(
            'EndCoordinateInMeters', 1e-3
        )
        self.numberOfSteps = self._settingsGroup.createIntegerParameter('NumberOfSteps', 100)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
