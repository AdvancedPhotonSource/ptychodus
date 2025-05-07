from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Probe')
        self._group.add_observer(self)

        self.builder = self._group.create_string_parameter('Builder', 'Disk')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/probe.npy'))
        self.file_type = self._group.create_string_parameter('FileType', 'NPY')

        self.num_incoherent_modes = self._group.create_integer_parameter(
            'NumberOfIncoherentModes', 1, minimum=1
        )
        self.orthogonalize_incoherent_modes = self._group.create_boolean_parameter(
            'OrthogonalizeIncoherentModes', True
        )
        self.incoherent_mode_decay_type = self._group.create_string_parameter(
            'IncoherentModeDecayType', 'Polynomial'
        )
        self.incoherent_mode_decay_ratio = self._group.create_real_parameter(
            'IncoherentModeDecayRatio', 1.0, minimum=0.0, maximum=1.0
        )
        self.num_coherent_modes = self._group.create_integer_parameter(
            'NumberOfCoherentModes', 1, minimum=1
        )

        self.disk_diameter_m = self._group.create_real_parameter(
            'DiskDiameterInMeters', 1e-6, minimum=0.0
        )
        self.rectangle_width_m = self._group.create_real_parameter(
            'RectangleWidthInMeters', 1e-6, minimum=0.0
        )
        self.rectangle_height_m = self._group.create_real_parameter(
            'RectangleHeightInMeters', 1e-6, minimum=0.0
        )

        self.super_gaussian_annular_radius_m = self._group.create_real_parameter(
            'SuperGaussianAnnularRadiusInMeters', 0, minimum=0.0
        )
        self.super_gaussian_width_m = self._group.create_real_parameter(
            'SuperGaussianWidthInMeters', 400e-6, minimum=0.0
        )
        self.super_gaussian_order_parameter = self._group.create_real_parameter(
            'SuperGaussianOrderParameter', 1, minimum=1.0
        )

        self.zone_plate_diameter_m = self._group.create_real_parameter(
            'ZonePlateDiameterInMeters', 180e-6, minimum=0.0
        )
        self.outermost_zone_width_m = self._group.create_real_parameter(
            'OutermostZoneWidthInMeters', 50e-9, minimum=0.0
        )
        self.central_beamstop_diameter_m = self._group.create_real_parameter(
            'CentralBeamstopDiameterInMeters', 60e-6, minimum=0.0
        )
        self.defocus_distance_m = self._group.create_real_parameter('DefocusDistanceInMeters', 0.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
