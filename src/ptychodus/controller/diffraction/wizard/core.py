import logging

from PyQt5.QtWidgets import QWizard

from ....api.diffraction import DiffractionMetadata
from ....model.diffraction import (
    DetectorSettings,
    DiffractionAPI,
    DiffractionDatasetRepository,
    DiffractionSettings,
)
from ....model.product import ProductSettings
from ....view.widgets import ExceptionDialog

from ...data import FileDialogFactory
from ..detector_extent import DetectorExtentSource
from .bad_pixels import OpenDatasetWizardBadPixelsViewController
from .files import OpenDatasetWizardFilesViewController
from .metadata import OpenDatasetWizardMetadataViewController
from .processing import OpenDatasetWizardProcessingViewController

logger = logging.getLogger(__name__)


class OpenDatasetWizardController:
    def __init__(
        self,
        settings: DiffractionSettings,
        detector_settings: DetectorSettings,
        product_settings: ProductSettings,
        extent_source: DetectorExtentSource,
        api: DiffractionAPI,
        repository: DiffractionDatasetRepository,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._api = api
        self._detector_settings = detector_settings
        self._repository = repository
        self._pending_dataset_index = -1
        self._file_view_controller = OpenDatasetWizardFilesViewController(
            settings, api, file_dialog_factory
        )
        self._metadata_view_controller = OpenDatasetWizardMetadataViewController(
            detector_settings,
            settings,
            product_settings,
            self._get_pending_metadata,
        )
        self._bad_pixels_view_controller = OpenDatasetWizardBadPixelsViewController(
            detector_settings, api, file_dialog_factory
        )
        self._processing_view_controller = OpenDatasetWizardProcessingViewController(
            settings, extent_source, file_dialog_factory
        )

        self._wizard = QWizard()
        self._wizard.setWindowTitle('Open Dataset')
        self._wizard.addPage(self._file_view_controller.get_widget())
        self._wizard.addPage(self._metadata_view_controller.get_widget())
        self._wizard.addPage(self._bad_pixels_view_controller.get_widget())
        self._wizard.addPage(self._processing_view_controller.get_widget())

        next_button = self._wizard.button(QWizard.WizardButton.NextButton)

        if next_button is None:
            raise ValueError('next_button is None!')
        else:
            next_button.clicked.connect(self._execute_next_button_action)

        finish_button = self._wizard.button(QWizard.WizardButton.FinishButton)

        if finish_button is None:
            raise ValueError('finish_button is None!')
        else:
            finish_button.clicked.connect(self._execute_finish_button_action)

    def _get_pending_metadata(self) -> DiffractionMetadata:
        if self._pending_dataset_index < 0:
            return DiffractionMetadata.create_null()
        return self._repository[self._pending_dataset_index].get_metadata()

    def _execute_next_button_action(self) -> None:
        # Handlers fire AFTER Qt has advanced the wizard, so currentPage() is the
        # page the user just arrived on. The branches below therefore describe
        # actions taken on the "leaving X → arriving Y" transition:
        page = self._wizard.currentPage()

        if page is self._metadata_view_controller.get_widget():
            # Files → Metadata: read the dataset file so metadata checkboxes
            # populate for the page that just became visible.
            self._pending_dataset_index = self._file_view_controller.open_dataset()
            self._metadata_view_controller.refresh()
        elif page is self._bad_pixels_view_controller.get_widget():
            # Metadata → Bad Pixels: apply the metadata-import selections. Seed
            # the bad-pixels file browser to focus on the current settings path
            # if it points to a valid file.
            self._metadata_view_controller.import_metadata()
            self._bad_pixels_view_controller.restart()
        elif page is self._processing_view_controller.get_widget():
            # Bad Pixels → Processing: load and apply the bad-pixels mask (if
            # any) to the pending dataset before the user configures processing.
            if self._pending_dataset_index >= 0:
                self._api.apply_bad_pixels(
                    self._pending_dataset_index,
                    self._detector_settings.bad_pixels_file_path.get_value(),
                    self._detector_settings.bad_pixels_file_type.get_value(),
                )

    def _execute_finish_button_action(self) -> None:
        if self._pending_dataset_index < 0:
            return
        try:
            self._api.load_all_arrays(dataset_index=self._pending_dataset_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Open Dataset', exc)

    def open_dataset(self) -> None:
        self._pending_dataset_index = -1
        self._wizard.restart()
        self._file_view_controller.restart()
        self._wizard.show()
