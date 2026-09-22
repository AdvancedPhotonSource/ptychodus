from collections.abc import Sequence

import numpy

from ptychodus.api.constants import energy_eV_to_J, energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider, compute_object_geometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.propagate import compute_far_field_pixel_geometry, compute_magnification
from ptychodus.api.probe_positions import ProbePosition

from .metadata import MetadataRepositoryItem
from .probe_positions import ProbePositionsRepositoryItem


class ProductGeometry(ProbeGeometryProvider, ObjectGeometryProvider, Observable, Observer):
    def __init__(
        self,
        metadata_item: MetadataRepositoryItem,
        scan_item: ProbePositionsRepositoryItem,
    ) -> None:
        super().__init__()
        self._metadata_item = metadata_item
        self._scan_item = scan_item
        # Set via set_detector_extent()/set_detector_pixel_geometry() when a dataset
        # is bound (see ProductRepositoryItem.bind_dataset / unbind_dataset). Both
        # describe the assembled patterns (post-preprocessing), so downstream
        # calculations can consume them directly without re-folding a live
        # DiffractionSettings preprocessing pipeline -- re-folding would double-apply
        # binning on re-import of an already-preprocessed dataset. Derived quantities
        # degrade to zero-sized while unbound.
        self._detector_extent: ImageExtent | None = None
        self._detector_pixel_geometry: PixelGeometry | None = None

        self._metadata_item.add_observer(self)
        self._scan_item.add_observer(self)

    def set_detector_extent(self, extent: ImageExtent | None) -> None:
        if extent == self._detector_extent:
            return
        self._detector_extent = extent
        self.notify_observers()

    def set_detector_pixel_geometry(self, geometry: PixelGeometry | None) -> None:
        if geometry == self._detector_pixel_geometry:
            return
        self._detector_pixel_geometry = geometry
        self.notify_observers()

    @property
    def probe_photon_count(self) -> float:
        return self._metadata_item.probe_photon_count.get_value()

    @property
    def probe_energy_J(self) -> float:  # noqa: N802
        return energy_eV_to_J(self._metadata_item.probe_energy_eV.get_value())

    @property
    def probe_wavelength_m(self) -> float:
        return energy_eV_to_wavelength_m(self._metadata_item.probe_energy_eV.get_value())

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
    def focus_object_distance_m(self) -> float:
        return self._metadata_item.focus_object_distance_m.get_value()

    @property
    def magnification(self) -> float:
        """See :func:`ptychodus.api.product.compute_magnification`."""
        return compute_magnification(self.detector_distance_m, self.focus_object_distance_m)

    @property
    def object_plane_propagation_distance_m(self) -> float:
        """Propagation distance of the equivalent parallel-beam geometry, ``z_d / M``.

        A cone beam magnifying by ``M`` images like a parallel beam propagating this
        much shorter distance onto pixels this much smaller, which is the pairing
        :meth:`get_object_plane_pixel_geometry` applies. Equals the detector distance
        whenever there is no focusing optic.
        """
        try:
            return self.detector_distance_m / self.magnification
        except ZeroDivisionError:
            return 0.0

    @property
    def _lambda_z_m2(self) -> float:
        return self.probe_wavelength_m * self.object_plane_propagation_distance_m

    def _get_detector_extent(self) -> ImageExtent:
        # No dataset bound yet: degrade to a zero-sized extent so downstream
        # divisions bail out gracefully (they already handle ZeroDivisionError).
        extent = self._detector_extent
        if extent is None:
            return ImageExtent(width_px=0, height_px=0)
        return extent

    def get_detector_pixel_geometry(self) -> PixelGeometry:
        # No dataset bound yet: degrade to a zero-sized geometry so downstream
        # divisions bail out gracefully (they already handle ZeroDivisionError).
        geometry = self._detector_pixel_geometry
        if geometry is None:
            return PixelGeometry(width_m=0.0, height_m=0.0)
        return geometry

    def get_object_plane_pixel_geometry(self) -> PixelGeometry:
        """Sample-plane sampling implied by the detector and the illumination geometry.

        With a focusing optic the cone beam projects the detector pixels onto the
        sample demagnified by ``M``; without one the sampling is the far-field
        reciprocal relation. Degrades to zero-sized on any degenerate geometry, as
        the rest of this class does.
        """
        magnification = self.magnification

        if magnification != 1.0:
            detector_pixel_geometry = self.get_detector_pixel_geometry()

            try:
                return PixelGeometry(
                    width_m=detector_pixel_geometry.width_m / magnification,
                    height_m=detector_pixel_geometry.height_m / magnification,
                )
            except ZeroDivisionError:
                return PixelGeometry(width_m=0.0, height_m=0.0)

        try:
            return compute_far_field_pixel_geometry(
                self.get_detector_pixel_geometry(),
                self._get_detector_extent(),
                wavelength_m=self.probe_wavelength_m,
                propagation_distance_m=self.detector_distance_m,
            )
        except ZeroDivisionError:
            return PixelGeometry(width_m=0.0, height_m=0.0)

    @property
    def fresnel_number(self) -> float:
        """Full-aperture Fresnel number ``W H / (lambda z)`` at the **object** plane.

        This is the propagation-regime indicator: much less than one is far field, near
        one is transitional, much greater than one is near field. The detector-plane
        aperture number is its reciprocal up to the pixel count -- ``Fr_detector *
        Fr_object == width_px * height_px`` exactly -- so reporting the detector plane
        would read large precisely when the geometry is deeply far field.

        Distinct from ``PropagatorParameters.pixel_fresnel_number``, which is the
        per-pixel quantity ``dx^2 / (lambda z)``.

        ``z`` is the equivalent parallel-beam distance
        :attr:`object_plane_propagation_distance_m`, so the indicator stays meaningful
        when a focusing optic magnifies the geometry.

        Degrades to 0.0 when no dataset is bound or the distance is zero. Without a
        focusing optic that is the correct limit at this plane: the object-plane width
        is ``lambda z / dx_d``, so ``W^2 / (lambda z) = lambda z / dx_d^2 -> 0`` as
        z -> 0. With one the object-plane width is fixed at ``N dx_d / M`` and the true
        limit diverges, so the zero is a degenerate-input guard rather than a limit.
        """
        extent = self._get_detector_extent()
        pixel_geometry = self.get_object_plane_pixel_geometry()
        width_m = extent.width_px * pixel_geometry.width_m
        height_m = extent.height_px * pixel_geometry.height_m
        area_m2 = width_m * height_m
        try:
            return area_m2 / self._lambda_z_m2
        except ZeroDivisionError:
            return 0.0

    @property
    def _detector_numerical_aperture_sq(self) -> float:
        extent = self._get_detector_extent()
        pixel_geometry = self.get_detector_pixel_geometry()
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
        extent = self._get_detector_extent()
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
        pixel_geometry = self.get_object_plane_pixel_geometry()

        if pixel_geometry.is_valid:
            try:
                return compute_object_geometry(self.get_probe_positions(), probe_geometry)
            except ValueError:
                pass  # Empty scan — fall through to the probe-sized default below.

        # Detector unbound or scan not yet loaded: degrade to a probe-sized canvas at
        # the origin so downstream UI has valid dimensions to render.
        return ObjectGeometry(
            width_px=probe_geometry.width_px if pixel_geometry.is_valid else 0,
            height_px=probe_geometry.height_px if pixel_geometry.is_valid else 0,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            center_x_m=0.0,
            center_y_m=0.0,
        )

    def is_object_geometry_valid(self, geometry: ObjectGeometry) -> bool:
        expected_geometry = self.get_object_geometry()
        return geometry.get_pixel_geometry().is_valid and geometry.contains(expected_geometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self.notify_observers()
        elif observable is self._scan_item:
            self.notify_observers()
