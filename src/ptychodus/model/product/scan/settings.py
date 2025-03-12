from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ScanSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Scan')
        self._settingsGroup.add_observer(self)

        self.builder = self._settingsGroup.create_string_parameter('Builder', 'rectangular_raster')
        self.filePath = self._settingsGroup.create_path_parameter(
            'FilePath', Path('/path/to/scan.csv')
        )
        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'CSV')

        self.affineTransformAX = self._settingsGroup.create_real_parameter('AffineTransformAX', 1.0)
        self.affineTransformAY = self._settingsGroup.create_real_parameter('AffineTransformAY', 0.0)
        self.affineTransformATInMeters = self._settingsGroup.create_real_parameter(
            'AffineTransformATInMeters', 0.0
        )

        self.affineTransformBX = self._settingsGroup.create_real_parameter('AffineTransformBX', 0.0)
        self.affineTransformBY = self._settingsGroup.create_real_parameter('AffineTransformBY', 1.0)
        self.affineTransformBTInMeters = self._settingsGroup.create_real_parameter(
            'AffineTransformBTInMeters', 0.0
        )
        self.jitterRadiusInMeters = self._settingsGroup.create_real_parameter(
            'JitterRadiusInMeters', 0.0, minimum=0.0
        )

        self.expandBoundingBox = self._settingsGroup.create_boolean_parameter(
            'ExpandBoundingBox', False
        )
        self.expandedBoundingBoxMinimumXInMeters = self._settingsGroup.create_real_parameter(
            'ExpandedBoundingBoxMinimumXInMeters', -5e-7
        )
        self.expandedBoundingBoxMaximumXInMeters = self._settingsGroup.create_real_parameter(
            'ExpandedBoundingBoxMaximumXInMeters', +5e-7
        )
        self.expandedBoundingBoxMinimumYInMeters = self._settingsGroup.create_real_parameter(
            'ExpandedBoundingBoxMinimumYInMeters', -5e-7
        )
        self.expandedBoundingBoxMaximumYInMeters = self._settingsGroup.create_real_parameter(
            'ExpandedBoundingBoxMaximumYInMeters', +5e-7
        )

        self.numberOfPointsX = self._settingsGroup.create_integer_parameter(
            'NumberOfPointsX', 10, minimum=0
        )
        self.numberOfPointsY = self._settingsGroup.create_integer_parameter(
            'NumberOfPointsY', 10, minimum=0
        )
        self.stepSizeXInMeters = self._settingsGroup.create_real_parameter(
            'StepSizeXInMeters', 1e-6, minimum=0.0
        )
        self.stepSizeYInMeters = self._settingsGroup.create_real_parameter(
            'StepSizeYInMeters', 1e-6, minimum=0.0
        )

        self.radialStepSizeInMeters = self._settingsGroup.create_real_parameter(
            'RadialStepSizeInMeters', 1e-6, minimum=0.0
        )
        self.numberOfShells = self._settingsGroup.create_integer_parameter(
            'NumberOfShells', 5, minimum=0
        )
        self.numberOfPointsInFirstShell = self._settingsGroup.create_integer_parameter(
            'NumberOfPointsInFirstShell', 10, minimum=0
        )

        self.amplitudeXInMeters = self._settingsGroup.create_real_parameter(
            'AmplitudeXInMeters', 4.5e-6, minimum=0.0
        )
        self.amplitudeYInMeters = self._settingsGroup.create_real_parameter(
            'AmplitudeYInMeters', 4.5e-6, minimum=0.0
        )
        self.angularStepXInTurns = self._settingsGroup.create_real_parameter(
            'AngularStepXInTurns', 0.03
        )
        self.angularStepYInTurns = self._settingsGroup.create_real_parameter(
            'AngularStepYInTurns', 0.04
        )
        self.angularShiftInTurns = self._settingsGroup.create_real_parameter(
            'AngularShiftInTurns', 0.25
        )

        self.radiusScalarInMeters = self._settingsGroup.create_real_parameter(
            'RadiusScalarInMeters', 0.5e-6, minimum=0.0
        )

    @property
    def numberOfPoints(self) -> int:
        return self.numberOfPointsX.get_value() * self.numberOfPointsY.get_value()

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
