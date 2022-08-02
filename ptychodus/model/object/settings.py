from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class ObjectSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Random')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'NPY')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/object.npy'))

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ObjectSettings:
        settings = cls(settingsRegistry.createGroup('Object'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
