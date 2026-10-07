from collections.abc import Sequence

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider, compute_object_geometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.propagate import ProductGeometry, compute_product_geometry
from ptychodus.api.probe_positions import ProbePosition

from .metadata import MetadataRepositoryItem
from .probe_positions import ProbePositionsRepositoryItem


class ProductGeometryProvider(ProbeGeometryProvider, ObjectGeometryProvider, Observable, Observer):
    """Live product geometry: the bound metadata, scan and detector, as a provider.

    Adapts a product under edit to the two geometry-provider interfaces, tracking
    changes to the metadata and scan it was built from. The derived quantities
    themselves are the frozen :class:`ProductGeometry` returned by
    :meth:`get_derived_values`; this class only supplies its inputs and republishes
    the handful of them the provider interfaces require.
    """

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

    def get_derived_values(self) -> ProductGeometry:
        """Beam and sampling quantities implied by the current metadata and detector.

        Recomputed per call rather than cached: it is a dozen float operations, and
        this class is :class:`Observable` precisely so that holders refresh after a
        change instead of reading a stale snapshot.
        """
        return compute_product_geometry(
            probe_energy_eV=self._metadata_item.probe_energy_eV.get_value(),
            probe_photon_count=self._metadata_item.probe_photon_count.get_value(),
            exposure_time_s=self._metadata_item.exposure_time_s.get_value(),
            detector_distance_m=self._metadata_item.detector_distance_m.get_value(),
            focus_object_distance_m=self._metadata_item.focus_object_distance_m.get_value(),
            far_field=self._metadata_item.far_field.get_value(),
            detector_extent=self._detector_extent,
            detector_pixel_geometry=self._detector_pixel_geometry,
        )

    @property
    def probe_photon_count(self) -> float:
        return self._metadata_item.probe_photon_count.get_value()

    @property
    def probe_wavelength_m(self) -> float:
        return energy_eV_to_wavelength_m(self._metadata_item.probe_energy_eV.get_value())

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return self.get_derived_values().probe_power_W

    @property
    def num_scan_points(self) -> int:
        return len(self._scan_item.get_probe_positions())

    @property
    def detector_distance_m(self) -> float:
        return self._metadata_item.detector_distance_m.get_value()

    @property
    def far_field(self) -> bool:
        return self._metadata_item.far_field.get_value()

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
        """Sample-plane sampling implied by the detector and the declared regime.

        See :attr:`ProductGeometry.object_plane_pixel_geometry`. Kept as a method
        because it is what :class:`ProbeGeometryProvider` consumers reach for, and
        because :meth:`get_probe_geometry` and :meth:`get_object_geometry` below both
        need it.
        """
        return self.get_derived_values().object_plane_pixel_geometry

    def get_probe_geometry(self) -> ProbeGeometry:
        extent = self._get_detector_extent()
        pixel_geometry = self.get_object_plane_pixel_geometry()
        return ProbeGeometry(
            width_px=extent.width_px,
            height_px=extent.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

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

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata_item:
            self.notify_observers()
        elif observable is self._scan_item:
            self.notify_observers()
