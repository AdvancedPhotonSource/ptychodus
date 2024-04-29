from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ObjectSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Object')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringEntry('Builder', 'Random')
        self.inputFilePath = self._settingsGroup.createPathEntry('InputFilePath',
                                                                 Path('/path/to/object.npy'))
        self.inputFileType = self._settingsGroup.createStringEntry('InputFileType', 'NPY')

        self.objectLayerDistanceInMeters = self._settingsGroup.createRealEntry(
            'ObjectLayerDistanceInMeters', '1e-6')

        self.extraPaddingX = self._settingsGroup.createIntegerEntry('ExtraPaddingX', 1)
        self.extraPaddingY = self._settingsGroup.createIntegerEntry('ExtraPaddingY', 1)
        self.amplitudeMean = self._settingsGroup.createRealEntry('AmplitudeMean', '0.5')
        self.amplitudeDeviation = self._settingsGroup.createRealEntry('AmplitudeDeviation', '0')
        self.phaseDeviation = self._settingsGroup.createRealEntry('PhaseDeviation', '0')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
