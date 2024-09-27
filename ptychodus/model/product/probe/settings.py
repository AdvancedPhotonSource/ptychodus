from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    PathParameter,
    RealParameter,
    StringParameter,
)


class ProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Probe")
        self._settingsGroup.addObserver(self)

        self.builder = StringParameter(self._settingsGroup, "Builder", "Disk")
        self.filePath = PathParameter(self._settingsGroup, "FilePath", Path("/path/to/probe.npy"))
        self.fileType = StringParameter(self._settingsGroup, "FileType", "NPY")

        self.numberOfModes = IntegerParameter(self._settingsGroup, "NumberOfModes", 1)
        self.orthogonalizeModesEnabled = BooleanParameter(
            self._settingsGroup, "OrthogonalizeModesEnabled", True
        )
        self.modeDecayType = StringParameter(self._settingsGroup, "ModeDecayType", "Polynomial")
        self.modeDecayRatio = RealParameter(self._settingsGroup, "ModeDecayRatio", 1.0)

        self.diskDiameterInMeters = RealParameter(self._settingsGroup, "DiskDiameterInMeters", 1e-6)
        self.rectangleWidthInMeters = RealParameter(
            self._settingsGroup, "RectangleWidthInMeters", 1e-6
        )
        self.rectangleHeightInMeters = RealParameter(
            self._settingsGroup, "RectangleHeightInMeters", 1e-6
        )

        self.superGaussianAnnularRadiusInMeters = RealParameter(
            self._settingsGroup, "SuperGaussianAnnularRadiusInMeters", 0
        )
        self.superGaussianWidthInMeters = RealParameter(
            self._settingsGroup, "SuperGaussianWidthInMeters", 400e-6
        )
        self.superGaussianOrderParameter = RealParameter(
            self._settingsGroup, "SuperGaussianOrderParameter", 1
        )

        self.zonePlateDiameterInMeters = RealParameter(
            self._settingsGroup, "ZonePlateDiameterInMeters", 180e-6
        )
        self.outermostZoneWidthInMeters = RealParameter(
            self._settingsGroup, "OutermostZoneWidthInMeters", 50e-9
        )
        self.centralBeamstopDiameterInMeters = RealParameter(
            self._settingsGroup, "CentralBeamstopDiameterInMeters", 60e-6
        )
        self.defocusDistanceInMeters = RealParameter(
            self._settingsGroup, "DefocusDistanceInMeters", 0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
