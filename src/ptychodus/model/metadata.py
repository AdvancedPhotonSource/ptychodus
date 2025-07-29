from __future__ import annotations

from ptychodus.api.observer import Observable
from ptychodus.api.diffraction import DiffractionMetadata

from .diffraction import (
    DetectorSettings,
    DiffractionDatasetObserver,
    AssembledDiffractionDataset,
    DiffractionSettings,
)
from .product import ProductSettings


class MetadataPresenter(Observable, DiffractionDatasetObserver):
    def __init__(
        self,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
        dataset: AssembledDiffractionDataset,
        product_settings: ProductSettings,
    ) -> None:
        super().__init__()
        self._detector_settings = detector_settings
        self._diffraction_settings = diffraction_settings
        self._dataset = dataset
        self._product_settings = product_settings

        dataset.add_observer(self)

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._dataset.get_metadata()

    def can_sync_detector_extent(self) -> bool:
        return self._metadata.detector_extent is not None

    def sync_detector_extent(self) -> None:
        detector_extent = self._metadata.detector_extent

        if detector_extent:
            self._detector_settings.width_px.set_value(detector_extent.width_px)
            self._detector_settings.height_px.set_value(detector_extent.height_px)

    def can_sync_detector_pixel_size(self) -> bool:
        return self._metadata.detector_pixel_geometry is not None

    def sync_detector_pixel_size(self) -> None:
        pixel_geometry = self._metadata.detector_pixel_geometry

        if pixel_geometry:
            self._detector_settings.pixel_width_m.set_value(pixel_geometry.width_m)
            self._detector_settings.pixel_height_m.set_value(pixel_geometry.height_m)

    def can_sync_detector_bit_depth(self) -> bool:
        return self._metadata.detector_bit_depth is not None

    def sync_detector_bit_depth(self) -> None:
        bit_depth = self._metadata.detector_bit_depth

        if bit_depth:
            self._detector_settings.bit_depth.set_value(bit_depth)

    def can_sync_pattern_crop_center(self) -> bool:
        return self._metadata.crop_center is not None or self._metadata.detector_extent is not None

    def can_sync_pattern_crop_extent(self) -> bool:
        return self._metadata.detector_extent is not None

    def sync_pattern_crop(self, sync_center: bool, sync_extent: bool) -> None:
        if sync_center:
            crop_center = self._metadata.crop_center

            if crop_center:
                self._diffraction_settings.crop_center_x_px.set_value(crop_center.position_x_px)
                self._diffraction_settings.crop_center_y_px.set_value(crop_center.position_y_px)
            elif self._metadata.detector_extent:
                self._diffraction_settings.crop_center_x_px.set_value(
                    int(self._metadata.detector_extent.width_px) // 2
                )
                self._diffraction_settings.crop_center_y_px.set_value(
                    int(self._metadata.detector_extent.height_px) // 2
                )

        if sync_extent and self._metadata.detector_extent:
            center_x = self._diffraction_settings.crop_center_x_px.get_value()
            center_y = self._diffraction_settings.crop_center_y_px.get_value()

            extent_x = int(self._metadata.detector_extent.width_px)
            extent_y = int(self._metadata.detector_extent.height_px)

            max_radius_x = min(center_x, extent_x - center_x)
            max_radius_y = min(center_y, extent_y - center_y)
            max_radius = min(max_radius_x, max_radius_y)
            crop_diameter = 1

            while crop_diameter < max_radius:
                crop_diameter <<= 1

            self._diffraction_settings.crop_width_px.set_value(crop_diameter)
            self._diffraction_settings.crop_height_px.set_value(crop_diameter)

    def can_sync_probe_energy(self) -> bool:
        return self._metadata.probe_energy_eV is not None

    def sync_probe_energy(self) -> None:
        energy_eV = self._metadata.probe_energy_eV  # noqa: N806

        if energy_eV:
            self._product_settings.probe_energy_eV.set_value(energy_eV)

    def can_sync_probe_photon_count(self) -> bool:
        return self._metadata.probe_photon_count is not None

    def sync_probe_photon_count(self) -> None:
        photon_count = self._metadata.probe_photon_count

        if photon_count:
            self._product_settings.probe_photon_count.set_value(photon_count)

    def can_sync_exposure_time(self) -> bool:
        return self._metadata.exposure_time_s is not None

    def sync_exposure_time(self) -> None:
        exposure_time_s = self._metadata.exposure_time_s

        if exposure_time_s:
            self._product_settings.exposure_time_s.set_value(exposure_time_s)

    def can_sync_detector_distance(self) -> bool:
        return self._metadata.detector_distance_m is not None

    def sync_detector_distance(self) -> None:
        distance_m = self._metadata.detector_distance_m

        if distance_m:
            self._product_settings.detector_distance_m.set_value(distance_m)

    def handle_bad_pixels_changed(self, num_bad_pixels: int) -> None:
        pass

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self.notify_observers()
