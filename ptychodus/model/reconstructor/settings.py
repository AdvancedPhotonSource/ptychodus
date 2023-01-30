from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsGroup, SettingsRegistry


class ReconstructorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.algorithm = settingsGroup.createStringEntry('Algorithm', 'Tike/rPIE')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ReconstructorSettings:
        settings = cls(settingsRegistry.createGroup('Reconstructor'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
