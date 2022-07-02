from pathlib import Path

from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup


class ScanSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Snake')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'CSV')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/scan.csv'))
        self.inputFileSeriesKey = settingsGroup.createStringEntry('InputFileSeriesKey', 'Default')
        self.extentX = settingsGroup.createIntegerEntry('ExtentX', 10)
        self.extentY = settingsGroup.createIntegerEntry('ExtentY', 10)
        self.stepSizeXInMeters = settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')
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
