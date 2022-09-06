from __future__ import annotations
from decimal import Decimal
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class ScanSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Snake')
        self.activeScan = settingsGroup.createStringEntry('ActiveScan', 'Snake')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'CSV')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/scan.csv'))
        self.numberOfPointsX = settingsGroup.createIntegerEntry('NumberOfPointsX', 10)
        self.numberOfPointsY = settingsGroup.createIntegerEntry('NumberOfPointsY', 10)
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
        self.stepSizeInTurns = settingsGroup.createRealEntry('StepSizeInTurns',
                                                             (3 - Decimal(5).sqrt()) / 2)
        self.phaseDifferenceInTurns = settingsGroup.createRealEntry('PhaseDifferenceInTurns', '0')
        self.radiusScalarInMeters = settingsGroup.createRealEntry('RadiusScalarInMeters', '1e-6')
        self.amplitudeXInMeters = settingsGroup.createRealEntry('AmplitudeXInMeters', '1e-3')
        self.amplitudeYInMeters = settingsGroup.createRealEntry('AmplitudeYInMeters', '1e-3')
        self.centroidXInMeters = settingsGroup.createRealEntry('CentroidXInMeters', '0')
        self.centroidYInMeters = settingsGroup.createRealEntry('CentroidYInMeters', '0')
        self.jitterRadiusInMeters = settingsGroup.createRealEntry('JitterRadiusInMeters', '0')
        self.transform = settingsGroup.createStringEntry('Transform', '+X+Y')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ScanSettings:
        settings = cls(settingsRegistry.createGroup('Scan'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
