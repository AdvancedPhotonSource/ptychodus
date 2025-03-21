from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Probe')
        self._settingsGroup.add_observer(self)

        self.builder = self._settingsGroup.create_string_parameter('Builder', 'Disk')
        self.filePath = self._settingsGroup.create_path_parameter(
            'FilePath', Path('/path/to/probe.npy')
        )
        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'NPY')

        self.numberOfCoherentModes = self._settingsGroup.create_integer_parameter(
            'NumberOfCoherentModes', 1, minimum=1
        )
        self.num_incoherent_modes = self._settingsGroup.create_integer_parameter(
            'NumberOfIncoherentModes', 1, minimum=1
        )
        self.orthogonalize_incoherent_modes = self._settingsGroup.create_boolean_parameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.incoherent_mode_decay_type = self._settingsGroup.create_string_parameter(
            'IncoherentModeDecayType', 'Polynomial'
        )
        self.incoherent_mode_decay_ratio = self._settingsGroup.create_real_parameter(
            'IncoherentModeDecayRatio', 1.0, minimum=0.0, maximum=1.0
        )

        self.diskDiameterInMeters = self._settingsGroup.create_real_parameter(
            'DiskDiameterInMeters', 1e-6, minimum=0.0
        )
        self.rectangleWidthInMeters = self._settingsGroup.create_real_parameter(
            'RectangleWidthInMeters', 1e-6, minimum=0.0
        )
        self.rectangleHeightInMeters = self._settingsGroup.create_real_parameter(
            'RectangleHeightInMeters', 1e-6, minimum=0.0
        )

        self.superGaussianAnnularRadiusInMeters = self._settingsGroup.create_real_parameter(
            'SuperGaussianAnnularRadiusInMeters', 0, minimum=0.0
        )
        self.superGaussianWidthInMeters = self._settingsGroup.create_real_parameter(
            'SuperGaussianWidthInMeters', 400e-6, minimum=0.0
        )
        self.superGaussianOrderParameter = self._settingsGroup.create_real_parameter(
            'SuperGaussianOrderParameter', 1, minimum=1.0
        )

        self.zonePlateDiameterInMeters = self._settingsGroup.create_real_parameter(
            'ZonePlateDiameterInMeters', 180e-6, minimum=0.0
        )
        self.outermostZoneWidthInMeters = self._settingsGroup.create_real_parameter(
            'OutermostZoneWidthInMeters', 50e-9, minimum=0.0
        )
        self.centralBeamstopDiameterInMeters = self._settingsGroup.create_real_parameter(
            'CentralBeamstopDiameterInMeters', 60e-6, minimum=0.0
        )
        self.defocusDistanceInMeters = self._settingsGroup.create_real_parameter(
            'DefocusDistanceInMeters', 0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
