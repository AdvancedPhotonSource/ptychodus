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
        # Signed beamline coordinate, so deliberately unbounded below: a negative
        # value places the focus upstream of the object. See compute_magnification.
        self.focus_object_distance_m = self._group.create_real_parameter(
            'FocusObjectDistanceInMeters', 0.0
        )
        self.photon_energy_eV = self._group.create_real_parameter(
            'PhotonEnergyInElectronVolts', 10000.0, minimum=0.0
        )
        self.probe_photon_count = self._group.create_real_parameter(
            'ProbePhotonCount', 0.0, minimum=0.0
        )
        self.exposure_time_s = self._group.create_real_parameter(
            'ExposureTimeInSeconds', 0.0, minimum=0.0
        )
        self.mass_attenuation_m2_per_kg = self._group.create_real_parameter(
            'MassAttenuationSquareMetersPerKilogram', 0.0, minimum=0.0
        )
        self.tomography_angle_deg = self._group.create_real_parameter(
            'TomographyAngleInDegrees', 0.0
        )
        self.tilt_angle_deg = self._group.create_real_parameter('TiltAngleInDegrees', 0.0)
        self.polarization = self._group.create_string_parameter('Polarization', '')
        self.far_field = self._group.create_boolean_parameter('FarField', True)
        # Zero means the I0 the probe positions carry is uncalibrated.
        self.i0_photons_per_count = self._group.create_real_parameter(
            'I0PhotonsPerCount', 0.0, minimum=0.0
        )

        # Measure the defocus from the parallax between bright-field images when a
        # product is created, which sets focus_object_distance_m above and so reaches
        # every probe builder through resolve_defocus_distance_m. Off by default: it
        # costs one binned pass over the patterns, and it only answers in far field.
        self.estimate_defocus_from_parallax = self._group.create_boolean_parameter(
            'EstimateDefocusFromParallax', False
        )
        self.parallax_bin_size_px = self._group.create_integer_parameter(
            'ParallaxBinSizeInPixels', 4, minimum=1
        )
        self.parallax_min_significance = self._group.create_real_parameter(
            'ParallaxMinSignificance', 3.0, minimum=0.0
        )
        self.parallax_min_defocus_dof = self._group.create_real_parameter(
            'ParallaxMinDefocusDof', 1.0, minimum=0.0
        )
        self.parallax_rtol = self._group.create_real_parameter('ParallaxRtol', 0.5, minimum=0.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
