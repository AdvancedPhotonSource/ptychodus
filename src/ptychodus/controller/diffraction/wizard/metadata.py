from PyQt5.QtWidgets import QWizardPage

from ptychodus.api.observer import Observable, Observer

from ....model.metadata import MetadataPresenter
from ....view.diffraction import OpenDatasetWizardMetadataPage


class OpenDatasetWizardMetadataViewController(Observer):
    def __init__(self, presenter: MetadataPresenter) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = OpenDatasetWizardMetadataPage()

        presenter.add_observer(self)
        self._sync_model_to_view()
        self._page._set_complete(True)

    def import_metadata(self) -> None:
        if self._page.detector_extent_check_box.isChecked():
            self._presenter.sync_detector_extent()

        if self._page.detector_pixel_size_check_box.isChecked():
            self._presenter.sync_detector_pixel_size()

        if self._page.detector_bit_depth_check_box.isChecked():
            self._presenter.sync_detector_bit_depth()

        if self._page.detector_distance_check_box.isChecked():
            self._presenter.sync_detector_distance()

        self._presenter.sync_pattern_crop(
            sync_center=self._page.pattern_crop_center_check_box.isChecked(),
            sync_extent=self._page.pattern_crop_extent_check_box.isChecked(),
        )

        if self._page.probe_energy_check_box.isChecked():
            self._presenter.sync_probe_energy()

        if self._page.probe_photon_count_check_box.isChecked():
            self._presenter.sync_probe_photon_count()

        if self._page.exposure_time_check_box.isChecked():
            self._presenter.sync_exposure_time()

    def _sync_model_to_view(self) -> None:
        can_sync_detector_extent = self._presenter.can_sync_detector_extent()
        self._page.detector_extent_check_box.setVisible(can_sync_detector_extent)
        self._page.detector_extent_check_box.setChecked(can_sync_detector_extent)

        can_sync_detector_pixel_size = self._presenter.can_sync_detector_pixel_size()
        self._page.detector_pixel_size_check_box.setVisible(can_sync_detector_pixel_size)
        self._page.detector_pixel_size_check_box.setChecked(can_sync_detector_pixel_size)

        can_sync_detector_bit_depth = self._presenter.can_sync_detector_bit_depth()
        self._page.detector_bit_depth_check_box.setVisible(can_sync_detector_bit_depth)
        self._page.detector_bit_depth_check_box.setChecked(can_sync_detector_bit_depth)

        can_sync_detector_distance = self._presenter.can_sync_detector_distance()
        self._page.detector_distance_check_box.setVisible(can_sync_detector_distance)
        self._page.detector_distance_check_box.setChecked(can_sync_detector_distance)

        can_sync_pattern_crop_center = self._presenter.can_sync_pattern_crop_center()
        self._page.pattern_crop_center_check_box.setVisible(can_sync_pattern_crop_center)
        self._page.pattern_crop_center_check_box.setChecked(can_sync_pattern_crop_center)

        can_sync_pattern_crop_extent = self._presenter.can_sync_pattern_crop_extent()
        self._page.pattern_crop_extent_check_box.setVisible(can_sync_pattern_crop_extent)
        self._page.pattern_crop_extent_check_box.setChecked(can_sync_pattern_crop_extent)

        can_sync_probe_energy = self._presenter.can_sync_probe_energy()
        self._page.probe_energy_check_box.setVisible(can_sync_probe_energy)
        self._page.probe_energy_check_box.setChecked(can_sync_probe_energy)

        can_sync_probe_photon_count = self._presenter.can_sync_probe_photon_count()
        self._page.probe_photon_count_check_box.setVisible(can_sync_probe_photon_count)
        self._page.probe_photon_count_check_box.setChecked(can_sync_probe_photon_count)

        can_sync_exposure_time = self._presenter.can_sync_exposure_time()
        self._page.exposure_time_check_box.setVisible(can_sync_exposure_time)
        self._page.exposure_time_check_box.setChecked(can_sync_exposure_time)

    def get_widget(self) -> QWizardPage:
        return self._page

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
