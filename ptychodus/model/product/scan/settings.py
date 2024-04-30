from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ScanSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Scan')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringEntry('Builder', 'rectangular_raster')
        self.inputFilePath = self._settingsGroup.createPathEntry('InputFilePath',
                                                                 Path('/path/to/scan.csv'))
        self.inputFileType = self._settingsGroup.createStringEntry('InputFileType', 'CSV')

        self.affineTransformAX = self._settingsGroup.createRealEntry('AffineTransformAX', '1')
        self.affineTransformAY = self._settingsGroup.createRealEntry('AffineTransformAY', '0')
        self.affineTransformATInMeters = self._settingsGroup.createRealEntry(
            'AffineTransformATInMeters', '0')

        self.affineTransformBX = self._settingsGroup.createRealEntry('AffineTransformBX', '0')
        self.affineTransformBY = self._settingsGroup.createRealEntry('AffineTransformBY', '1')
        self.affineTransformBTInMeters = self._settingsGroup.createRealEntry(
            'AffineTransformBTInMeters', '0')
        self.jitterRadiusInMeters = self._settingsGroup.createRealEntry(
            'JitterRadiusInMeters', '0')

        self.expandBoundingBox = self._settingsGroup.createBooleanEntry('ExpandBoundingBox', False)
        self.expandedBoundingBoxMinimumXInMeters = self._settingsGroup.createRealEntry(
            'ExpandedBoundingBoxMinimumXInMeters', '-5e-7')
        self.expandedBoundingBoxMaximumXInMeters = self._settingsGroup.createRealEntry(
            'ExpandedBoundingBoxMaximumXInMeters', '+5e-7')
        self.expandedBoundingBoxMinimumYInMeters = self._settingsGroup.createRealEntry(
            'ExpandedBoundingBoxMinimumYInMeters', '-5e-7')
        self.expandedBoundingBoxMaximumYInMeters = self._settingsGroup.createRealEntry(
            'ExpandedBoundingBoxMaximumYInMeters', '+5e-7')

        self.numberOfPointsX = self._settingsGroup.createIntegerEntry('NumberOfPointsX', 10)
        self.numberOfPointsY = self._settingsGroup.createIntegerEntry('NumberOfPointsY', 10)
        self.stepSizeXInMeters = self._settingsGroup.createRealEntry('StepSizeXInMeters', '1e-6')
        self.stepSizeYInMeters = self._settingsGroup.createRealEntry('StepSizeYInMeters', '1e-6')

        self.radialStepSizeInMeters = self._settingsGroup.createRealEntry(
            'RadialStepSizeInMeters', '1e-6')
        self.numberOfShells = self._settingsGroup.createIntegerEntry('NumberOfShells', 5)
        self.numberOfPointsInFirstShell = self._settingsGroup.createIntegerEntry(
            'NumberOfPointsInFirstShell', 10)

        self.amplitudeXInMeters = self._settingsGroup.createRealEntry(
            'AmplitudeXInMeters', '4.5e-6')
        self.amplitudeYInMeters = self._settingsGroup.createRealEntry(
            'AmplitudeYInMeters', '4.5e-6')
        self.angularStepXInTurns = self._settingsGroup.createRealEntry(
            'AngularStepXInTurns', '0.03')
        self.angularStepYInTurns = self._settingsGroup.createRealEntry(
            'AngularStepYInTurns', '0.04')
        self.angularShiftInTurns = self._settingsGroup.createRealEntry(
            'AngularShiftInTurns', '0.25')

        self.radiusScalarInMeters = self._settingsGroup.createRealEntry(
            'RadiusScalarInMeters', '0.5e-6')

    @property
    def numberOfPoints(self) -> int:
        return self.numberOfPointsX.value * self.numberOfPointsY.value

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
