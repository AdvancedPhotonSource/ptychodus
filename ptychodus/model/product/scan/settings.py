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


class ScanSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Scan")
        self._settingsGroup.addObserver(self)

        self.builder = StringParameter(self._settingsGroup, "Builder", "rectangular_raster")
        self.filePath = PathParameter(self._settingsGroup, "FilePath", Path("/path/to/scan.csv"))
        self.fileType = StringParameter(self._settingsGroup, "FileType", "CSV")

        self.affineTransformAX = RealParameter(self._settingsGroup, "AffineTransformAX", 1.0)
        self.affineTransformAY = RealParameter(self._settingsGroup, "AffineTransformAY", 0.0)
        self.affineTransformATInMeters = RealParameter(
            self._settingsGroup, "AffineTransformATInMeters", 0.0
        )

        self.affineTransformBX = RealParameter(self._settingsGroup, "AffineTransformBX", 0.0)
        self.affineTransformBY = RealParameter(self._settingsGroup, "AffineTransformBY", 1.0)
        self.affineTransformBTInMeters = RealParameter(
            self._settingsGroup, "AffineTransformBTInMeters", 0.0
        )
        self.jitterRadiusInMeters = RealParameter(self._settingsGroup, "JitterRadiusInMeters", 0.0)

        self.expandBoundingBox = BooleanParameter(self._settingsGroup, "ExpandBoundingBox", False)
        self.expandedBoundingBoxMinimumXInMeters = RealParameter(
            self._settingsGroup, "ExpandedBoundingBoxMinimumXInMeters", -5e-7
        )
        self.expandedBoundingBoxMaximumXInMeters = RealParameter(
            self._settingsGroup, "ExpandedBoundingBoxMaximumXInMeters", +5e-7
        )
        self.expandedBoundingBoxMinimumYInMeters = RealParameter(
            self._settingsGroup, "ExpandedBoundingBoxMinimumYInMeters", -5e-7
        )
        self.expandedBoundingBoxMaximumYInMeters = RealParameter(
            self._settingsGroup, "ExpandedBoundingBoxMaximumYInMeters", +5e-7
        )

        self.numberOfPointsX = IntegerParameter(self._settingsGroup, "NumberOfPointsX", 10)
        self.numberOfPointsY = IntegerParameter(self._settingsGroup, "NumberOfPointsY", 10)
        self.stepSizeXInMeters = RealParameter(self._settingsGroup, "StepSizeXInMeters", 1e-6)
        self.stepSizeYInMeters = RealParameter(self._settingsGroup, "StepSizeYInMeters", 1e-6)

        self.radialStepSizeInMeters = RealParameter(
            self._settingsGroup, "RadialStepSizeInMeters", 1e-6
        )
        self.numberOfShells = IntegerParameter(self._settingsGroup, "NumberOfShells", 5)
        self.numberOfPointsInFirstShell = IntegerParameter(
            self._settingsGroup, "NumberOfPointsInFirstShell", 10
        )

        self.amplitudeXInMeters = RealParameter(self._settingsGroup, "AmplitudeXInMeters", 4.5e-6)
        self.amplitudeYInMeters = RealParameter(self._settingsGroup, "AmplitudeYInMeters", 4.5e-6)
        self.angularStepXInTurns = RealParameter(self._settingsGroup, "AngularStepXInTurns", 0.03)
        self.angularStepYInTurns = RealParameter(self._settingsGroup, "AngularStepYInTurns", 0.04)
        self.angularShiftInTurns = RealParameter(self._settingsGroup, "AngularShiftInTurns", 0.25)

        self.radiusScalarInMeters = RealParameter(
            self._settingsGroup, "RadiusScalarInMeters", 0.5e-6
        )

    @property
    def numberOfPoints(self) -> int:
        return self.numberOfPointsX.getValue() * self.numberOfPointsY.getValue()

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
