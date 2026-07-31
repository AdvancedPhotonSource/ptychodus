from __future__ import annotations

from ptychodus.api.diffraction import DiffractionMetadata

from .diffraction import DetectorSettings, DiffractionSettings
from .product import ProductSettings


class MetadataPresenter:
    """Stateless helper for syncing DiffractionMetadata fields into settings.

    Every method takes an explicit :class:`DiffractionMetadata` argument; the
    presenter holds no notion of a currently-selected dataset. Callers (UI
    controllers) look up the metadata for the dataset they care about and pass
    it in.
    """

    def __init__(
        self,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
        product_settings: ProductSettings,
    ) -> None:
        self._detector_settings = detector_settings
        self._diffraction_settings = diffraction_settings
        self._product_settings = product_settings

    def can_sync_detector_extent(self, metadata: DiffractionMetadata) -> bool:
        return metadata.detector_extent is not None

    def sync_detector_extent(self, metadata: DiffractionMetadata) -> None:
        detector_extent = metadata.detector_extent

        if detector_extent:
            self._detector_settings.width_px.set_value(detector_extent.width_px)
            self._detector_settings.height_px.set_value(detector_extent.height_px)

    def can_sync_detector_pixel_size(self, metadata: DiffractionMetadata) -> bool:
        return metadata.detector_pixel_geometry is not None

    def sync_detector_pixel_size(self, metadata: DiffractionMetadata) -> None:
        pixel_geometry = metadata.detector_pixel_geometry

        if pixel_geometry:
            self._detector_settings.pixel_width_m.set_value(pixel_geometry.width_m)
            self._detector_settings.pixel_height_m.set_value(pixel_geometry.height_m)

    def can_sync_pattern_crop_center(self, metadata: DiffractionMetadata) -> bool:
        return metadata.crop_center is not None or metadata.detector_extent is not None

    def can_sync_pattern_crop_extent(self, metadata: DiffractionMetadata) -> bool:
        return metadata.detector_extent is not None

    def sync_pattern_crop(
        self, metadata: DiffractionMetadata, *, sync_center: bool, sync_extent: bool
    ) -> None:
        if sync_center:
            crop_center = metadata.crop_center

            if crop_center:
                self._diffraction_settings.crop_center_x_px.set_value(crop_center.position_x_px)
                self._diffraction_settings.crop_center_y_px.set_value(crop_center.position_y_px)
            elif metadata.detector_extent:
                self._diffraction_settings.crop_center_x_px.set_value(
                    int(metadata.detector_extent.width_px) // 2
                )
                self._diffraction_settings.crop_center_y_px.set_value(
                    int(metadata.detector_extent.height_px) // 2
                )

        if sync_extent and metadata.detector_extent:
            center_x = self._diffraction_settings.crop_center_x_px.get_value()
            center_y = self._diffraction_settings.crop_center_y_px.get_value()

            extent_x = int(metadata.detector_extent.width_px)
            extent_y = int(metadata.detector_extent.height_px)

            max_radius_x = min(center_x, extent_x - center_x)
            max_radius_y = min(center_y, extent_y - center_y)
            max_radius = min(max_radius_x, max_radius_y)
            crop_diameter = 1

            while crop_diameter < max_radius:
                crop_diameter <<= 1

            self._diffraction_settings.crop_width_px.set_value(crop_diameter)
            self._diffraction_settings.crop_height_px.set_value(crop_diameter)

    def can_sync_probe_energy(self, metadata: DiffractionMetadata) -> bool:
        return metadata.probe_energy_eV is not None

    def sync_probe_energy(self, metadata: DiffractionMetadata) -> None:
        energy_eV = metadata.probe_energy_eV  # noqa: N806

        if energy_eV:
            self._product_settings.probe_energy_eV.set_value(energy_eV)

    def can_sync_probe_photon_count(self, metadata: DiffractionMetadata) -> bool:
        return metadata.probe_photon_count is not None

    def sync_probe_photon_count(self, metadata: DiffractionMetadata) -> None:
        photon_count = metadata.probe_photon_count

        if photon_count:
            self._product_settings.probe_photon_count.set_value(photon_count)

    def can_sync_exposure_time(self, metadata: DiffractionMetadata) -> bool:
        return metadata.exposure_time_s is not None

    def sync_exposure_time(self, metadata: DiffractionMetadata) -> None:
        exposure_time_s = metadata.exposure_time_s

        if exposure_time_s:
            self._product_settings.exposure_time_s.set_value(exposure_time_s)

    def can_sync_detector_distance(self, metadata: DiffractionMetadata) -> bool:
        return metadata.detector_distance_m is not None

    def sync_detector_distance(self, metadata: DiffractionMetadata) -> None:
        distance_m = metadata.detector_distance_m

        if distance_m:
            self._product_settings.detector_distance_m.set_value(distance_m)
