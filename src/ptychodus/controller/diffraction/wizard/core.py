import logging

from PyQt5.QtWidgets import QWizard

from ....api.diffraction import DiffractionMetadata
from ....model.metadata import MetadataPresenter
from ....model.diffraction import (
    DetectorSettings,
    DiffractionAPI,
    DiffractionDatasetRepository,
    DiffractionSettings,
)
from ....view.widgets import ExceptionDialog

from ...data import FileDialogFactory
from ..detector_extent import DetectorExtentSource
from .files import OpenDatasetWizardFilesViewController
from .metadata import OpenDatasetWizardMetadataViewController
from .patterns import OpenDatasetWizardPatternsViewController

logger = logging.getLogger(__name__)


class OpenDatasetWizardController:
    def __init__(
        self,
        settings: DiffractionSettings,
        detector_settings: DetectorSettings,
        extent_source: DetectorExtentSource,
        api: DiffractionAPI,
        repository: DiffractionDatasetRepository,
        metadata_presenter: MetadataPresenter,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._api = api
        self._repository = repository
        self._pending_dataset_index = -1
        self._file_view_controller = OpenDatasetWizardFilesViewController(
            settings, detector_settings, api, file_dialog_factory
        )
        self._metadata_view_controller = OpenDatasetWizardMetadataViewController(
            metadata_presenter, self._get_pending_metadata
        )
        self._patterns_view_controller = OpenDatasetWizardPatternsViewController(
            settings, extent_source, file_dialog_factory
        )

        self._wizard = QWizard()
        self._wizard.setWindowTitle('Open Dataset')
        self._wizard.addPage(self._file_view_controller.get_widget())
        self._wizard.addPage(self._metadata_view_controller.get_widget())
        self._wizard.addPage(self._patterns_view_controller.get_widget())

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
        page = self._wizard.currentPage()

        if page is self._metadata_view_controller.get_widget():
            self._pending_dataset_index = self._file_view_controller.open_dataset()
            self._metadata_view_controller.refresh()
        elif page is self._patterns_view_controller.get_widget():
            self._metadata_view_controller.import_metadata()

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
