from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class ScanSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.builder = settingsGroup.createStringEntry('Builder', 'Snake')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/scan.csv'))
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'CSV')
        self.amplitudeXInMeters = settingsGroup.createRealEntry('AmplitudeXInMeters', '4.5e-6')
        self.amplitudeYInMeters = settingsGroup.createRealEntry('AmplitudeYInMeters', '4.5e-6')
        self.angularShiftInTurns = settingsGroup.createRealEntry('AngularShiftInTurns', '0.25')
        self.angularStepXInTurns = settingsGroup.createRealEntry('AngularStepXInTurns', '0.03')
        self.angularStepYInTurns = settingsGroup.createRealEntry('AngularStepYInTurns', '0.04')
        self.overrideCentroidXEnabled = settingsGroup.createBooleanEntry(
            'OverrideCentroidXEnabled', False)
        self.overrideCentroidXInMeters = settingsGroup.createRealEntry(
            'OverrideCentroidXInMeters', '0')
        self.overrideCentroidYEnabled = settingsGroup.createBooleanEntry(
            'OverrideCentroidYEnabled', False)
        self.overrideCentroidYInMeters = settingsGroup.createRealEntry(
            'OverrideCentroidYInMeters', '0')
        self.jitterRadiusInMeters = settingsGroup.createRealEntry('JitterRadiusInMeters', '0')
        self.numberOfPointsX = settingsGroup.createIntegerEntry('NumberOfPointsX', 10)
        self.numberOfPointsY = settingsGroup.createIntegerEntry('NumberOfPointsY', 10)
        self.radiusScalarInMeters = settingsGroup.createRealEntry('RadiusScalarInMeters', '0.5e-6')
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
        self.radialStepSizeInMeters = settingsGroup.createRealEntry('RadialStepSizeInMeters',
                                                                    '1e-6')
        self.numberOfShells = settingsGroup.createIntegerEntry('NumberOfShells', 5)
        self.numberOfPointsInFirstShell = settingsGroup.createIntegerEntry(
            'NumberOfPointsInFirstShell', 10)
        self.transform = settingsGroup.createStringEntry('Transform', '+X+Y')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ScanSettings:
        settingsGroup = settingsRegistry.createGroup('Scan')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
