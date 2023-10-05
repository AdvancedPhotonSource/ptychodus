from __future__ import annotations

from ...api.observer import Observable, Observer
from ...api.settings import SettingsGroup, SettingsRegistry


class ReconstructorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.algorithm = settingsGroup.createStringEntry('Algorithm', 'Tike/lstsq_grad')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ReconstructorSettings:
        settingsGroup = settingsRegistry.createGroup('Reconstructor')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
