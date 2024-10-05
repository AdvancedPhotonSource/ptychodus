from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ScanSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Scan')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringParameter('Builder', 'rectangular_raster')
        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/scan.csv')
        )
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'CSV')

        self.affineTransformAX = self._settingsGroup.createRealParameter('AffineTransformAX', 1.0)
        self.affineTransformAY = self._settingsGroup.createRealParameter('AffineTransformAY', 0.0)
        self.affineTransformATInMeters = self._settingsGroup.createRealParameter(
            'AffineTransformATInMeters', 0.0
        )

        self.affineTransformBX = self._settingsGroup.createRealParameter('AffineTransformBX', 0.0)
        self.affineTransformBY = self._settingsGroup.createRealParameter('AffineTransformBY', 1.0)
        self.affineTransformBTInMeters = self._settingsGroup.createRealParameter(
            'AffineTransformBTInMeters', 0.0
        )
        self.jitterRadiusInMeters = self._settingsGroup.createRealParameter(
            'JitterRadiusInMeters', 0.0, minimum=0.0
        )

        self.expandBoundingBox = self._settingsGroup.createBooleanParameter(
            'ExpandBoundingBox', False
        )
        self.expandedBoundingBoxMinimumXInMeters = self._settingsGroup.createRealParameter(
            'ExpandedBoundingBoxMinimumXInMeters', -5e-7
        )
        self.expandedBoundingBoxMaximumXInMeters = self._settingsGroup.createRealParameter(
            'ExpandedBoundingBoxMaximumXInMeters', +5e-7
        )
        self.expandedBoundingBoxMinimumYInMeters = self._settingsGroup.createRealParameter(
            'ExpandedBoundingBoxMinimumYInMeters', -5e-7
        )
        self.expandedBoundingBoxMaximumYInMeters = self._settingsGroup.createRealParameter(
            'ExpandedBoundingBoxMaximumYInMeters', +5e-7
        )

        self.numberOfPointsX = self._settingsGroup.createIntegerParameter(
            'NumberOfPointsX', 10, minimum=0
        )
        self.numberOfPointsY = self._settingsGroup.createIntegerParameter(
            'NumberOfPointsY', 10, minimum=0
        )
        self.stepSizeXInMeters = self._settingsGroup.createRealParameter(
            'StepSizeXInMeters', 1e-6, minimum=0.0
        )
        self.stepSizeYInMeters = self._settingsGroup.createRealParameter(
            'StepSizeYInMeters', 1e-6, minimum=0.0
        )

        self.radialStepSizeInMeters = self._settingsGroup.createRealParameter(
            'RadialStepSizeInMeters', 1e-6, minimum=0.0
        )
        self.numberOfShells = self._settingsGroup.createIntegerParameter(
            'NumberOfShells', 5, minimum=0
        )
        self.numberOfPointsInFirstShell = self._settingsGroup.createIntegerParameter(
            'NumberOfPointsInFirstShell', 10, minimum=0
        )

        self.amplitudeXInMeters = self._settingsGroup.createRealParameter(
            'AmplitudeXInMeters', 4.5e-6, minimum=0.0
        )
        self.amplitudeYInMeters = self._settingsGroup.createRealParameter(
            'AmplitudeYInMeters', 4.5e-6, minimum=0.0
        )
        self.angularStepXInTurns = self._settingsGroup.createRealParameter(
            'AngularStepXInTurns', 0.03
        )
        self.angularStepYInTurns = self._settingsGroup.createRealParameter(
            'AngularStepYInTurns', 0.04
        )
        self.angularShiftInTurns = self._settingsGroup.createRealParameter(
            'AngularShiftInTurns', 0.25
        )

        self.radiusScalarInMeters = self._settingsGroup.createRealParameter(
            'RadiusScalarInMeters', 0.5e-6, minimum=0.0
        )

    @property
    def numberOfPoints(self) -> int:
        return self.numberOfPointsX.getValue() * self.numberOfPointsY.getValue()

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
