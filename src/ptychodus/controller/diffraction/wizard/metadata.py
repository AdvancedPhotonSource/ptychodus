from collections.abc import Callable

from PyQt5.QtWidgets import QWizardPage

from ptychodus.api.diffraction import DiffractionMetadata

from ....model.metadata import MetadataPresenter
from ....view.diffraction import OpenDatasetWizardMetadataPage


class OpenDatasetWizardMetadataViewController:
    def __init__(
        self,
        presenter: MetadataPresenter,
        get_metadata: Callable[[], DiffractionMetadata],
    ) -> None:
        self._presenter = presenter
        self._get_metadata = get_metadata
        self._page = OpenDatasetWizardMetadataPage()

        self.refresh()
        self._page._set_complete(True)

    def import_metadata(self) -> None:
        metadata = self._get_metadata()

        if self._page.detector_extent_check_box.isChecked():
            self._presenter.sync_detector_extent(metadata)

        if self._page.detector_pixel_size_check_box.isChecked():
            self._presenter.sync_detector_pixel_size(metadata)

        if self._page.detector_distance_check_box.isChecked():
            self._presenter.sync_detector_distance(metadata)

        self._presenter.sync_pattern_crop(
            metadata,
            sync_center=self._page.pattern_crop_center_check_box.isChecked(),
            sync_extent=self._page.pattern_crop_extent_check_box.isChecked(),
        )

        if self._page.probe_energy_check_box.isChecked():
            self._presenter.sync_probe_energy(metadata)

        if self._page.probe_photon_count_check_box.isChecked():
            self._presenter.sync_probe_photon_count(metadata)

        if self._page.exposure_time_check_box.isChecked():
            self._presenter.sync_exposure_time(metadata)

    def refresh(self) -> None:
        metadata = self._get_metadata()

        can_sync_detector_extent = self._presenter.can_sync_detector_extent(metadata)
        self._page.detector_extent_check_box.setVisible(can_sync_detector_extent)
        self._page.detector_extent_check_box.setChecked(can_sync_detector_extent)

        can_sync_detector_pixel_size = self._presenter.can_sync_detector_pixel_size(metadata)
        self._page.detector_pixel_size_check_box.setVisible(can_sync_detector_pixel_size)
        self._page.detector_pixel_size_check_box.setChecked(can_sync_detector_pixel_size)

        can_sync_detector_distance = self._presenter.can_sync_detector_distance(metadata)
        self._page.detector_distance_check_box.setVisible(can_sync_detector_distance)
        self._page.detector_distance_check_box.setChecked(can_sync_detector_distance)

        can_sync_pattern_crop_center = self._presenter.can_sync_pattern_crop_center(metadata)
        self._page.pattern_crop_center_check_box.setVisible(can_sync_pattern_crop_center)
        self._page.pattern_crop_center_check_box.setChecked(can_sync_pattern_crop_center)

        can_sync_pattern_crop_extent = self._presenter.can_sync_pattern_crop_extent(metadata)
        self._page.pattern_crop_extent_check_box.setVisible(can_sync_pattern_crop_extent)
        self._page.pattern_crop_extent_check_box.setChecked(can_sync_pattern_crop_extent)

        can_sync_probe_energy = self._presenter.can_sync_probe_energy(metadata)
        self._page.probe_energy_check_box.setVisible(can_sync_probe_energy)
        self._page.probe_energy_check_box.setChecked(can_sync_probe_energy)

        can_sync_probe_photon_count = self._presenter.can_sync_probe_photon_count(metadata)
        self._page.probe_photon_count_check_box.setVisible(can_sync_probe_photon_count)
        self._page.probe_photon_count_check_box.setChecked(can_sync_probe_photon_count)

        can_sync_exposure_time = self._presenter.can_sync_exposure_time(metadata)
        self._page.exposure_time_check_box.setVisible(can_sync_exposure_time)
        self._page.exposure_time_check_box.setChecked(can_sync_exposure_time)

    def get_widget(self) -> QWizardPage:
        return self._page
