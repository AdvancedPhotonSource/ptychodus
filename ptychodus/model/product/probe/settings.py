from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbeSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Probe')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringEntry('Builder', 'Disk')
        self.filePath = self._settingsGroup.createPathEntry('FilePath', Path('/path/to/probe.npy'))
        self.fileType = self._settingsGroup.createStringEntry('FileType', 'NPY')

        self.numberOfModes = self._settingsGroup.createIntegerEntry('NumberOfModes', 1)
        self.orthogonalizeModesEnabled = self._settingsGroup.createBooleanEntry(
            'OrthogonalizeModesEnabled', True)
        self.modeDecayType = self._settingsGroup.createStringEntry('ModeDecayType', 'Polynomial')
        self.modeDecayRatio = self._settingsGroup.createRealEntry('ModeDecayRatio', '1')

        self.diskDiameterInMeters = self._settingsGroup.createRealEntry(
            'DiskDiameterInMeters', '1e-6')
        self.rectangleWidthInMeters = self._settingsGroup.createRealEntry(
            'RectangleWidthInMeters', '1e-6')
        self.rectangleHeightInMeters = self._settingsGroup.createRealEntry(
            'RectangleHeightInMeters', '1e-6')

        self.superGaussianAnnularRadiusInMeters = self._settingsGroup.createRealEntry(
            'SuperGaussianAnnularRadiusInMeters', '0')
        self.superGaussianWidthInMeters = self._settingsGroup.createRealEntry(
            'SuperGaussianWidthInMeters', '400e-6')
        self.superGaussianOrderParameter = self._settingsGroup.createRealEntry(
            'SuperGaussianOrderParameter', '1')

        self.zonePlateDiameterInMeters = self._settingsGroup.createRealEntry(
            'ZonePlateDiameterInMeters', '180e-6')
        self.outermostZoneWidthInMeters = self._settingsGroup.createRealEntry(
            'OutermostZoneWidthInMeters', '50e-9')
        self.centralBeamstopDiameterInMeters = self._settingsGroup.createRealEntry(
            'CentralBeamstopDiameterInMeters', '60e-6')
        self.defocusDistanceInMeters = self._settingsGroup.createRealEntry(
            'DefocusDistanceInMeters', '0')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
