from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProductSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Products')
        self._group.add_observer(self)

        self.name = self._group.create_string_parameter('Name', 'Unnamed')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/product.h5'))
        self.file_type = self._group.create_string_parameter('FileType', 'HDF5')
        self.detector_distance_m = self._group.create_real_parameter(
            'DetectorDistanceInMeters', 1.0, minimum=0.0
        )
        self.probe_energy_eV = self._group.create_real_parameter(
            'ProbeEnergyInElectronVolts', 10000.0, minimum=0.0
        )
        self.probe_photon_count = self._group.create_real_parameter(
            'ProbePhotonCount', 0.0, minimum=0.0
        )
        self.exposure_time_s = self._group.create_real_parameter(
            'ExposureTimeInSeconds', 0.0, minimum=0.0
        )
        self.mass_attenuation_m2_kg = self._group.create_real_parameter(
            'MassAttenuationSquareMetersPerKilogram', 0.0, minimum=0.0
        )
        self.tomography_angle_deg = self._group.create_real_parameter(
            'TomographyAngleInDegrees', 0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
