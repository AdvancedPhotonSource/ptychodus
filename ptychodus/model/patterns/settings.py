from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class DiffractionDatasetSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.fileType = settingsGroup.createStringEntry('FileType', 'HDF5')
        self.filePath = settingsGroup.createPathEntry('FilePath', Path('/path/to/data.h5'))
        self.memmapEnabled = settingsGroup.createBooleanEntry('MemmapEnabled', False)
        self.scratchDirectory = settingsGroup.createPathEntry('ScratchDirectory',
                                                              Path.home() / '.ptychodus')
        self.numberOfDataThreads = settingsGroup.createIntegerEntry('NumberOfDataThreads', 8)
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '1')
        self.probeEnergyInElectronVolts = settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '10000')
        self.probePhotonsPerSecond = settingsGroup.createRealEntry('ProbePhotonsPerSecond', '0')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DiffractionDatasetSettings:
        settings = cls(settingsRegistry.createGroup('Diffraction Dataset'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class DiffractionPatternSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', True)
        self.cropCenterXInPixels = settingsGroup.createIntegerEntry('CropCenterXInPixels', 32)
        self.cropCenterYInPixels = settingsGroup.createIntegerEntry('CropCenterYInPixels', 32)
        self.cropWidthInPixels = settingsGroup.createIntegerEntry('CropWidthInPixels', 64)
        self.cropHeightInPixels = settingsGroup.createIntegerEntry('CropHeightInPixels', 64)
        self.flipXEnabled = settingsGroup.createBooleanEntry('FlipXEnabled', False)
        self.flipYEnabled = settingsGroup.createBooleanEntry('FlipYEnabled', False)
        self.valueLowerBoundEnabled = settingsGroup.createBooleanEntry(
            'ValueLowerBoundEnabled', False)
        self.valueLowerBound = settingsGroup.createIntegerEntry('ValueLowerBound', 0)
        self.valueUpperBoundEnabled = settingsGroup.createBooleanEntry(
            'ValueUpperBoundEnabled', False)
        self.valueUpperBound = settingsGroup.createIntegerEntry('ValueUpperBound', 65535)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DiffractionPatternSettings:
        settings = cls(settingsRegistry.createGroup('Diffraction Pattern'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
