from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PatternSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Patterns')
        self._settingsGroup.addObserver(self)

        self.fileType = self._settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = self._settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.memmapEnabled = self._settingsGroup.createBooleanEntry('MemmapEnabled', False)
        self.scratchDirectory = self._settingsGroup.createPathEntry('ScratchDirectory',
                                                                    Path.home() / '.ptychodus')
        self.numberOfDataThreads = self._settingsGroup.createIntegerEntry('NumberOfDataThreads', 8)

        self.cropEnabled = self._settingsGroup.createBooleanEntry('CropEnabled', True)
        self.cropCenterXInPixels = self._settingsGroup.createIntegerEntry(
            'CropCenterXInPixels', 32)
        self.cropCenterYInPixels = self._settingsGroup.createIntegerEntry(
            'CropCenterYInPixels', 32)
        self.cropWidthInPixels = self._settingsGroup.createIntegerEntry('CropWidthInPixels', 64)
        self.cropHeightInPixels = self._settingsGroup.createIntegerEntry('CropHeightInPixels', 64)
        self.flipXEnabled = self._settingsGroup.createBooleanEntry('FlipXEnabled', False)
        self.flipYEnabled = self._settingsGroup.createBooleanEntry('FlipYEnabled', False)
        self.valueLowerBoundEnabled = self._settingsGroup.createBooleanEntry(
            'ValueLowerBoundEnabled', False)
        self.valueLowerBound = self._settingsGroup.createIntegerEntry('ValueLowerBound', 0)
        self.valueUpperBoundEnabled = self._settingsGroup.createBooleanEntry(
            'ValueUpperBoundEnabled', False)
        self.valueUpperBound = self._settingsGroup.createIntegerEntry('ValueUpperBound', 65535)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ProductSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Products')
        self._settingsGroup.addObserver(self)

        self.detectorDistanceInMeters = self._settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '1')
        self.probeEnergyInElectronVolts = self._settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '10000')
        self.probePhotonsPerSecond = self._settingsGroup.createRealEntry(
            'ProbePhotonsPerSecond', '0')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
