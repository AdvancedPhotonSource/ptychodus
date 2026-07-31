from collections.abc import Sequence

import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
)
from ptychodus.api.probe_positions import ProbePosition

from ..diffraction import PatternSizer
from .metadata import MetadataRepositoryItem
from .probe_positions import ProbePositionsRepositoryItem


class ProductGeometry(ProbeGeometryProvider, ObjectGeometryProvider, Observable, Observer):
    def __init__(
        self,
        pattern_sizer: PatternSizer,
        metadata_item: MetadataRepositoryItem,
        scan_item: ProbePositionsRepositoryItem,
    ) -> None:
        super().__init__()
        self._pattern_sizer = pattern_sizer
        self._metadata_item = metadata_item
        self._scan_item = scan_item
        # Set via set_detector_extent() when a dataset is bound (see
        # ProductRepositoryItem.set_dataset). Derived quantities that need an extent
        # degenerate to zero-sized while unbound.
        self._detector_extent: ImageExtent | None = None

        self._pattern_sizer.add_observer(self)
        self._metadata_item.add_observer(self)
        self._scan_item.add_observer(self)

    def set_detector_extent(self, extent: ImageExtent | None) -> None:
        if extent == self._detector_extent:
            return
        self._detector_extent = extent
        self.notify_observers()

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
        return len(self._scan_item.get_probe_positions())

    @property
    def detector_distance_m(self) -> float:
        return self._metadata_item.detector_distance_m.get_value()

    @property
    def _lambda_z_m2(self) -> float:
        return self.probe_wavelength_m * self.detector_distance_m

    def get_detector_pixel_geometry(self):
        return self._pattern_sizer.get_processed_pixel_geometry()

    def get_object_plane_pixel_geometry(self) -> PixelGeometry:
        extent = self._pattern_sizer.get_processed_image_extent(self._detector_extent)
        detector_pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
        lambda_z = self._lambda_z_m2
        try:
            return PixelGeometry(
                width_m=lambda_z / (extent.width_px * detector_pixel_geometry.width_m),
                height_m=lambda_z / (extent.height_px * detector_pixel_geometry.height_m),
            )
        except ZeroDivisionError:
            return PixelGeometry(width_m=0.0, height_m=0.0)

    @property
    def fresnel_number(self) -> float:
        extent = self._pattern_sizer.get_processed_image_extent(self._detector_extent)
        pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
        width_m = extent.width_px * pixel_geometry.width_m
        height_m = extent.height_px * pixel_geometry.height_m
        area_m2 = width_m * height_m
        try:
            return area_m2 / self._lambda_z_m2
        except ZeroDivisionError:
            return 0.0

    @property
    def _detector_numerical_aperture_sq(self) -> float:
        extent = self._pattern_sizer.get_processed_image_extent(self._detector_extent)
        pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
        try:
            two_z_m = 2 * self.detector_distance_m
            NA_x = (extent.width_px * pixel_geometry.width_m) / two_z_m  # noqa: N806
            NA_y = (extent.height_px * pixel_geometry.height_m) / two_z_m  # noqa: N806
        except ZeroDivisionError:
            return 0.0
        return NA_x * NA_y

    @property
    def detector_numerical_aperture(self) -> float:
        return numpy.sqrt(self._detector_numerical_aperture_sq)

    @property
    def depth_of_field_m(self) -> float:
        return self.probe_wavelength_m / self._detector_numerical_aperture_sq

    def get_probe_geometry(self) -> ProbeGeometry:
        extent = self._pattern_sizer.get_processed_image_extent(self._detector_extent)
        pixel_geometry = self.get_object_plane_pixel_geometry()
        return ProbeGeometry(
            width_px=extent.width_px,
            height_px=extent.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    def is_probe_geometry_valid(self, geometry: ProbeGeometry) -> bool:
        expected = self.get_probe_geometry()
        if not geometry.get_pixel_geometry().is_valid:
            return False
        return geometry.width_m == expected.width_m and geometry.height_m == expected.height_m

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return self._scan_item.get_probe_positions()

    def get_object_geometry(self) -> ObjectGeometry:
        probe_geometry = self.get_probe_geometry()
        width_m = probe_geometry.width_m
        height_m = probe_geometry.height_m
        center_x_m = 0.0
        center_y_m = 0.0

        scan_bbox = self._scan_item.get_geometry()

        if scan_bbox is not None:
            width_m += scan_bbox.width_m
            height_m += scan_bbox.height_m
            center_x_m = scan_bbox.center_x_m
            center_y_m = scan_bbox.center_y_m

        pixel_geometry = self.get_object_plane_pixel_geometry()
        if pixel_geometry.is_valid:
            width_px = width_m / pixel_geometry.width_m
            height_px = height_m / pixel_geometry.height_m
        else:
            width_px = 0.0
            height_px = 0.0

        return ObjectGeometry(
            width_px=int(numpy.ceil(width_px)),
            height_px=int(numpy.ceil(height_px)),
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            center_x_m=center_x_m,
            center_y_m=center_y_m,
        )

    def is_object_geometry_valid(self, geometry: ObjectGeometry) -> bool:
        expected_geometry = self.get_object_geometry()
        return geometry.get_pixel_geometry().is_valid and geometry.contains(expected_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self.notify_observers()
        elif observable is self._scan_item:
            self.notify_observers()
        elif observable is self._pattern_sizer:
            self.notify_observers()
