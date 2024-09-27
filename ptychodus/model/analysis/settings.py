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


class ProbePropagationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("ProbePropagation")
        self._settingsGroup.addObserver(self)

        self.beginCoordinateInMeters = DecimalParameter(
            self._settingsGroup, "BeginCoordinateInMeters", "-1e-3"
        )
        self.endCoordinateInMeters = DecimalParameter(
            self._settingsGroup, "EndCoordinateInMeters", "+1e-3"
        )
        self.numberOfSteps = IntegerParameter(self._settingsGroup, "NumberOfSteps", 100)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class FluorescenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Fluorescence")
        self._settingsGroup.addObserver(self)

        self.filePath = PathParameter(self._settingsGroup, "FilePath", Path("/path/to/dataset.h5"))
        self.fileType = StringParameter(self._settingsGroup, "FileType", "XRF-Maps")
        self.useVSPI = BooleanParameter(self._settingsGroup, "UseVSPI", True)
        self.upscalingStrategy = StringParameter(self._settingsGroup, "UpscalingStrategy", "Linear")
        self.deconvolutionStrategy = StringParameter(
            self._settingsGroup, "DeconvolutionStrategy", "Richardson-Lucy"
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
