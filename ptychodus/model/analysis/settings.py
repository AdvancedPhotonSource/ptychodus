from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbePropagationSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('ProbePropagation')
        self._settingsGroup.addObserver(self)

        self.beginCoordinateInMeters = self._settingsGroup.createRealEntry(
            'BeginCoordinateInMeters', '-1e-3')
        self.endCoordinateInMeters = self._settingsGroup.createRealEntry(
            'EndCoordinateInMeters', '+1e-3')
        self.numberOfSteps = self._settingsGroup.createIntegerEntry('NumberOfSteps', 100)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class FluorescenceSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Fluorescence')
        self._settingsGroup.addObserver(self)

        self.filePath = self._settingsGroup.createPathEntry('FilePath',
                                                            Path('/path/to/dataset.h5'))
        self.fileType = self._settingsGroup.createStringEntry('FileType', 'XRF-Maps')
        self.useVSPI = self._settingsGroup.createBooleanEntry('UseVSPI', True)
        self.upscalingStrategy = self._settingsGroup.createStringEntry(
            'UpscalingStrategy', 'Linear')
        self.deconvolutionStrategy = self._settingsGroup.createStringEntry(
            'DeconvolutionStrategy', 'Richardson-Lucy')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
