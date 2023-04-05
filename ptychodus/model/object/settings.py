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
        self.amplitudeMean = settingsGroup.createRealEntry('AmplitudeMean', '0.5')
        self.amplitudeDeviation = settingsGroup.createRealEntry('AmplitudeDeviation', '0')
        self.randomizePhase = settingsGroup.createBooleanEntry('RandomizePhase', False)
        self.extraPaddingX = settingsGroup.createIntegerEntry('ExtraPaddingX', 0)
        self.extraPaddingY = settingsGroup.createIntegerEntry('ExtraPaddingY', 0)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ObjectSettings:
        settingsGroup = settingsRegistry.createGroup('Object')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
