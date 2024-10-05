from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Probe')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringParameter('Builder', 'Disk')
        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/probe.npy')
        )
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'NPY')

        self.numberOfModes = self._settingsGroup.createIntegerParameter(
            'NumberOfModes', 1, minimum=1
        )
        self.isOrthogonalizeModesEnabled = self._settingsGroup.createBooleanParameter(
            'OrthogonalizeModesEnabled', True
        )
        self.modeDecayType = self._settingsGroup.createStringParameter(
            'ModeDecayType', 'Polynomial'
        )
        self.modeDecayRatio = self._settingsGroup.createRealParameter(
            'ModeDecayRatio', 1.0, minimum=0.0, maximum=1.0
        )

        self.diskDiameterInMeters = self._settingsGroup.createRealParameter(
            'DiskDiameterInMeters', 1e-6, minimum=0.0
        )
        self.rectangleWidthInMeters = self._settingsGroup.createRealParameter(
            'RectangleWidthInMeters', 1e-6, minimum=0.0
        )
        self.rectangleHeightInMeters = self._settingsGroup.createRealParameter(
            'RectangleHeightInMeters', 1e-6, minimum=0.0
        )

        self.superGaussianAnnularRadiusInMeters = self._settingsGroup.createRealParameter(
            'SuperGaussianAnnularRadiusInMeters', 0, minimum=0.0
        )
        self.superGaussianWidthInMeters = self._settingsGroup.createRealParameter(
            'SuperGaussianWidthInMeters', 400e-6, minimum=0.0
        )
        self.superGaussianOrderParameter = self._settingsGroup.createRealParameter(
            'SuperGaussianOrderParameter', 1, minimum=1.0
        )

        self.zonePlateDiameterInMeters = self._settingsGroup.createRealParameter(
            'ZonePlateDiameterInMeters', 180e-6, minimum=0.0
        )
        self.outermostZoneWidthInMeters = self._settingsGroup.createRealParameter(
            'OutermostZoneWidthInMeters', 50e-9, minimum=0.0
        )
        self.centralBeamstopDiameterInMeters = self._settingsGroup.createRealParameter(
            'CentralBeamstopDiameterInMeters', 60e-6, minimum=0.0
        )
        self.defocusDistanceInMeters = self._settingsGroup.createRealParameter(
            'DefocusDistanceInMeters', 0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
