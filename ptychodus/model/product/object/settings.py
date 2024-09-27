from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.parametric import (
    IntegerParameter,
    PathParameter,
    RealParameter,
    StringParameter,
)


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Object")
        self._settingsGroup.addObserver(self)

        self.builder = StringParameter(self._settingsGroup, "Builder", "Random")
        self.filePath = PathParameter(self._settingsGroup, "FilePath", Path("/path/to/object.npy"))
        self.fileType = StringParameter(self._settingsGroup, "FileType", "NPY")

        self.objectLayerDistanceInMeters = RealParameter(
            self._settingsGroup, "ObjectLayerDistanceInMeters", 1e-6
        )

        self.extraPaddingX = IntegerParameter(self._settingsGroup, "ExtraPaddingX", 1)
        self.extraPaddingY = IntegerParameter(self._settingsGroup, "ExtraPaddingY", 1)
        self.amplitudeMean = RealParameter(self._settingsGroup, "AmplitudeMean", 0.5)
        self.amplitudeDeviation = RealParameter(self._settingsGroup, "AmplitudeDeviation", 0.0)
        self.phaseDeviation = RealParameter(self._settingsGroup, "PhaseDeviation", 0.0)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
