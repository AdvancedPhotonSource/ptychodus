from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ReconstructorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Reconstructor')
        self._settingsGroup.addObserver(self)

        self.algorithm = self._settingsGroup.createStringParameter('Algorithm', 'Tike/lstsq_grad')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
