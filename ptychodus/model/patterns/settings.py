from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.parametric import (
    BooleanParameter,
    DecimalParameter,
    IntegerParameter,
    PathParameter,
    StringParameter,
)


class PatternSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Patterns")
        self._settingsGroup.addObserver(self)

        self.fileType = StringParameter(self._settingsGroup, "FileType", "HDF5")
        self.filePath = PathParameter(self._settingsGroup, "FilePath", Path("/path/to/data.h5"))
        self.memmapEnabled = BooleanParameter(self._settingsGroup, "MemmapEnabled", False)
        self.scratchDirectory = PathParameter(self._settingsGroup, "ScratchDirectory",
                                              Path.home() / ".ptychodus")
        self.numberOfDataThreads = IntegerParameter(self._settingsGroup, "NumberOfDataThreads", 8)

        self.cropEnabled = BooleanParameter(self._settingsGroup, "CropEnabled", True)
        self.cropCenterXInPixels = IntegerParameter(self._settingsGroup, "CropCenterXInPixels", 32)
        self.cropCenterYInPixels = IntegerParameter(self._settingsGroup, "CropCenterYInPixels", 32)
        self.cropWidthInPixels = IntegerParameter(self._settingsGroup, "CropWidthInPixels", 64)
        self.cropHeightInPixels = IntegerParameter(self._settingsGroup, "CropHeightInPixels", 64)
        self.flipXEnabled = BooleanParameter(self._settingsGroup, "FlipXEnabled", False)
        self.flipYEnabled = BooleanParameter(self._settingsGroup, "FlipYEnabled", False)
        self.valueLowerBoundEnabled = BooleanParameter(self._settingsGroup,
                                                       "ValueLowerBoundEnabled", False)
        self.valueLowerBound = IntegerParameter(self._settingsGroup, "ValueLowerBound", 0)
        self.valueUpperBoundEnabled = BooleanParameter(self._settingsGroup,
                                                       "ValueUpperBoundEnabled", False)
        self.valueUpperBound = IntegerParameter(self._settingsGroup, "ValueUpperBound", 65535)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ProductSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Products")
        self._settingsGroup.addObserver(self)

        self.fileType = StringParameter(self._settingsGroup, "FileType", "HDF5")
        self.detectorDistanceInMeters = DecimalParameter(self._settingsGroup,
                                                         "DetectorDistanceInMeters", "1")
        self.probeEnergyInElectronVolts = DecimalParameter(self._settingsGroup,
                                                           "ProbeEnergyInElectronVolts", "10000")
        self.probePhotonsPerSecond = DecimalParameter(self._settingsGroup, "ProbePhotonsPerSecond",
                                                      "0")
        self.exposureTimeInSeconds = DecimalParameter(self._settingsGroup, "ExposureTimeInSeconds",
                                                      "0")

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
