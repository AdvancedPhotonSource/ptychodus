from PyQt5.QtWidgets import QFormLayout, QWizardPage

from ptychodus.api.observer import Observable, Observer

from ....model.diffraction import DetectorSettings, DiffractionAPI
from ....view.diffraction import OpenDatasetWizardBadPixelsPage
from ...data import FileDialogFactory
from .files import (
    OpenDatasetWizardBreadcrumbsViewController,
    OpenDatasetWizardFilePathViewController,
    OpenDatasetWizardFileTypeViewController,
    OpenDatasetWizardLocationViewController,
)


class OpenDatasetWizardBadPixelsViewController(Observer):
    """Bad-pixel file chooser — mirrors the Files page layout.

    Binds to :class:`DetectorSettings` for cross-session persistence (same
    convention as :class:`DiffractionSettings.file_path` for the Files page).
    The selection is applied by the wizard's next-button dispatcher on the
    Bad Pixels → Processing transition via
    :meth:`DiffractionAPI.apply_bad_pixels`.

    The page is always complete: leaving the file path invalid or unset is a
    supported "no bad-pixel mask" workflow.
    """

    def __init__(
        self,
        detector_settings: DetectorSettings,
        api: DiffractionAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._detector_settings = detector_settings
        self._file_dialog_factory = file_dialog_factory

        self._breadcrumbs_view_controller = OpenDatasetWizardBreadcrumbsViewController(
            file_dialog_factory
        )
        self._location_view_controller = OpenDatasetWizardLocationViewController(
            detector_settings.bad_pixels_file_path, file_dialog_factory
        )
        self._file_path_view_controller = OpenDatasetWizardFilePathViewController(
            detector_settings.bad_pixels_file_path, file_dialog_factory
        )
        self._file_type_view_controller = OpenDatasetWizardFileTypeViewController(
            api.get_bad_pixels_file_reader_parameter()
        )
        self._file_type_view_controller.get_parameter().add_observer(self)

        layout = QFormLayout()
        layout.addRow(self._breadcrumbs_view_controller.get_widget())
        layout.addRow('Location:', self._location_view_controller.get_widget())
        layout.addRow(self._file_path_view_controller.get_widget())
        layout.addRow('File Type:', self._file_type_view_controller.get_widget())

        self._page = OpenDatasetWizardBadPixelsPage()
        self._page.setTitle('Choose Bad Pixels File')
        self._page.setLayout(layout)

        self._handle_file_type_changed()

    def get_widget(self) -> QWizardPage:
        return self._page

    def restart(self) -> None:
        """Focus the file dialog on the current settings path if it points to a file."""
        current = self._detector_settings.bad_pixels_file_path.get_value()
        if current.exists():
            self._file_dialog_factory.set_open_working_directory(current)
        self._handle_file_type_changed()

    def _handle_file_type_changed(self) -> None:
        name_filters = self._file_type_view_controller.get_name_filters()
        self._file_path_view_controller.set_name_filters(name_filters)

    def _update(self, observable: Observable) -> None:
        if observable is self._file_type_view_controller.get_parameter():
            self._handle_file_type_changed()
