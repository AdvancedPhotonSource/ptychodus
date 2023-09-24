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
        self.numberOfSlices = settingsGroup.createIntegerEntry('NumberOfSlices', 1)
        self.amplitudeMean = settingsGroup.createRealEntry('AmplitudeMean', '0.5')
        self.amplitudeDeviation = settingsGroup.createRealEntry('AmplitudeDeviation', '0')
        self.phaseDeviation = settingsGroup.createRealEntry('PhaseDeviation', '0')
        self.extraPaddingX = settingsGroup.createIntegerEntry('ExtraPaddingX', 1)
        self.extraPaddingY = settingsGroup.createIntegerEntry('ExtraPaddingY', 1)
        self.phaseCenteringStrategy = settingsGroup.createStringEntry(
            'PhaseCenteringStrategy', 'Identity')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ObjectSettings:
        settingsGroup = settingsRegistry.createGroup('Object')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
