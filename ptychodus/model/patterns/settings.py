from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PatternSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Patterns')
        self._settingsGroup.addObserver(self)

        self.fileType = self._settingsGroup.createStringParameter('FileType', 'HDF5')
        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/data.h5')
        )
        self.memmapEnabled = self._settingsGroup.createBooleanParameter('MemmapEnabled', False)
        self.scratchDirectory = self._settingsGroup.createPathParameter(
            'ScratchDirectory', Path.home() / '.ptychodus'
        )
        self.numberOfDataThreads = self._settingsGroup.createIntegerParameter(
            'NumberOfDataThreads', 8
        )

        self.cropEnabled = self._settingsGroup.createBooleanParameter('CropEnabled', True)
        self.cropCenterXInPixels = self._settingsGroup.createIntegerParameter(
            'CropCenterXInPixels', 32
        )
        self.cropCenterYInPixels = self._settingsGroup.createIntegerParameter(
            'CropCenterYInPixels', 32
        )
        self.cropWidthInPixels = self._settingsGroup.createIntegerParameter('CropWidthInPixels', 64)
        self.cropHeightInPixels = self._settingsGroup.createIntegerParameter(
            'CropHeightInPixels', 64
        )
        self.flipXEnabled = self._settingsGroup.createBooleanParameter('FlipXEnabled', False)
        self.flipYEnabled = self._settingsGroup.createBooleanParameter('FlipYEnabled', False)
        self.valueLowerBoundEnabled = self._settingsGroup.createBooleanParameter(
            'ValueLowerBoundEnabled', False
        )
        self.valueLowerBound = self._settingsGroup.createIntegerParameter('ValueLowerBound', 0)
        self.valueUpperBoundEnabled = self._settingsGroup.createBooleanParameter(
            'ValueUpperBoundEnabled', False
        )
        self.valueUpperBound = self._settingsGroup.createIntegerParameter('ValueUpperBound', 65535)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ProductSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Products')
        self._settingsGroup.addObserver(self)

        self.name = self._settingsGroup.createStringParameter('Name', 'Unnamed')
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'HDF5')
        self.detectorDistanceInMeters = self._settingsGroup.createRealParameter(
            'DetectorDistanceInMeters', 1.0, minimum=0.0
        )
        self.probeEnergyInElectronVolts = self._settingsGroup.createRealParameter(
            'ProbeEnergyInElectronVolts', 10000.0, minimum=0.0
        )
        self.probePhotonsPerSecond = self._settingsGroup.createRealParameter(
            'ProbePhotonsPerSecond', 0.0, minimum=0.0
        )
        self.exposureTimeInSeconds = self._settingsGroup.createRealParameter(
            'ExposureTimeInSeconds', 0.0, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
