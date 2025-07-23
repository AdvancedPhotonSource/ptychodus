import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
)

from ..diffraction import PatternSizer
from .metadata import MetadataRepositoryItem
from .scan import ScanRepositoryItem


class ProductGeometry(ProbeGeometryProvider, ObjectGeometryProvider, Observable, Observer):
    def __init__(
        self,
        pattern_sizer: PatternSizer,
        metadata_item: MetadataRepositoryItem,
        scan_item: ScanRepositoryItem,
    ) -> None:
        super().__init__()
        self._pattern_sizer = pattern_sizer
        self._metadata_item = metadata_item
        self._scan_item = scan_item

        self._pattern_sizer.add_observer(self)
        self._metadata_item.add_observer(self)
        self._scan_item.add_observer(self)

    @property
    def probe_photon_count(self) -> float:
        return self._metadata_item.probe_photon_count.get_value()

    @property
    def probe_energy_J(self) -> float:  # noqa: N802
        return self._metadata_item.probe_energy_eV.get_value() * ELECTRON_VOLT_J

    @property
    def probe_wavelength_m(self) -> float:
        hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S  # noqa: N806

        try:
            return hc_Jm / self.probe_energy_J
        except ZeroDivisionError:
            return 0.0

    @property
    def probe_wavelengths_per_m(self) -> float:
        """wavenumber"""
        return 1.0 / self.probe_wavelength_m

    @property
    def probe_radians_per_m(self) -> float:
        """angular wavenumber"""
        return 2.0 * numpy.pi / self.probe_wavelength_m

    @property
    def probe_photons_per_s(self) -> float:
        try:
            return self.probe_photon_count / self._metadata_item.exposure_time_s.get_value()
        except ZeroDivisionError:
            return 0.0

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return self.probe_energy_J * self.probe_photons_per_s

    @property
    def num_scan_points(self) -> int:
        return len(self._scan_item.get_scan())

    @property
    def detector_distance_m(self) -> float:
        return self._metadata_item.detector_distance_m.get_value()

    @property
    def _lambda_z_m2(self) -> float:
        return self.probe_wavelength_m * self.detector_distance_m

    @property
    def object_plane_pixel_width_m(self) -> float:
        return self._lambda_z_m2 / self._pattern_sizer.get_processed_width_m()

    @property
    def object_plane_pixel_height_m(self) -> float:
        return self._lambda_z_m2 / self._pattern_sizer.get_processed_height_m()

    def get_detector_pixel_geometry(self):
        return self._pattern_sizer.get_processed_pixel_geometry()

    def get_object_plane_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.object_plane_pixel_width_m,
            height_m=self.object_plane_pixel_height_m,
        )

    @property
    def fresnel_number(self) -> float:
        width_m = self._pattern_sizer.get_processed_width_m()
        height_m = self._pattern_sizer.get_processed_height_m()
        area_m2 = width_m * height_m
        return area_m2 / self._lambda_z_m2

    @property
    def _detector_numerical_aperture_sq(self) -> float:
        two_z_m = 2 * self.detector_distance_m
        NA_x = self._pattern_sizer.get_processed_width_m() / two_z_m  # noqa: N806
        NA_y = self._pattern_sizer.get_processed_height_m() / two_z_m  # noqa: N806
        return NA_x * NA_y

    @property
    def detector_numerical_aperture(self) -> float:
        return numpy.sqrt(self._detector_numerical_aperture_sq)

    @property
    def depth_of_field_m(self) -> float:
        return self.probe_wavelength_m / self._detector_numerical_aperture_sq

    def get_probe_geometry(self) -> ProbeGeometry:
        extent = self._pattern_sizer.get_processed_image_extent()
        return ProbeGeometry(
            width_px=extent.width_px,
            height_px=extent.height_px,
            pixel_width_m=self.object_plane_pixel_width_m,
            pixel_height_m=self.object_plane_pixel_height_m,
        )

    def is_probe_geometry_valid(self, geometry: ProbeGeometry) -> bool:
        expected = self.get_probe_geometry()
        width_is_valid = geometry.pixel_width_m > 0.0 and geometry.width_m == expected.width_m
        height_is_valid = geometry.pixel_height_m > 0.0 and geometry.height_m == expected.height_m
        return width_is_valid and height_is_valid

    def get_object_geometry(self) -> ObjectGeometry:
        probe_geometry = self.get_probe_geometry()
        width_m = probe_geometry.width_m
        height_m = probe_geometry.height_m
        center_x_m = 0.0
        center_y_m = 0.0

        scan_bbox = self._scan_item.get_bounding_box()

        if scan_bbox is not None:
            width_m += scan_bbox.width_m
            height_m += scan_bbox.height_m
            center_x_m = scan_bbox.center_x_m
            center_y_m = scan_bbox.center_y_m

        width_px = width_m / self.object_plane_pixel_width_m
        height_px = height_m / self.object_plane_pixel_height_m

        return ObjectGeometry(
            width_px=int(numpy.ceil(width_px)),
            height_px=int(numpy.ceil(height_px)),
            pixel_width_m=self.object_plane_pixel_width_m,
            pixel_height_m=self.object_plane_pixel_height_m,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
        )

    def is_object_geometry_valid(self, geometry: ObjectGeometry) -> bool:
        expected_geometry = self.get_object_geometry()
        pixel_size_is_valid = geometry.pixel_width_m > 0.0 and geometry.pixel_height_m > 0.0
        return pixel_size_is_valid and geometry.contains(expected_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self.notify_observers()
        elif observable is self._scan_item:
            self.notify_observers()
        elif observable is self._pattern_sizer:
            self.notify_observers()
