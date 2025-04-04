import logging

from PyQt5.QtWidgets import QWizard

from ....model.metadata import MetadataPresenter
from ....model.patterns import PatternSettings, PatternSizer, PatternsAPI

from ...data import FileDialogFactory
from .files import OpenDatasetWizardFilesViewController
from .metadata import OpenDatasetWizardMetadataViewController
from .patterns import OpenDatasetWizardPatternsViewController

logger = logging.getLogger(__name__)


class OpenDatasetWizardController:
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
        api: PatternsAPI,
        metadata_presenter: MetadataPresenter,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._api = api
        self._file_view_controller = OpenDatasetWizardFilesViewController(
            settings, api, file_dialog_factory
        )
        self._metadata_view_controller = OpenDatasetWizardMetadataViewController(metadata_presenter)
        self._patterns_view_controller = OpenDatasetWizardPatternsViewController(
            settings, sizer, file_dialog_factory
        )

        self._wizard = QWizard()
        self._wizard.setWindowTitle('Open Dataset')
        self._wizard.addPage(self._file_view_controller.get_widget())
        self._wizard.addPage(self._metadata_view_controller.get_widget())
        self._wizard.addPage(self._patterns_view_controller.get_widget())

        self._wizard.button(QWizard.WizardButton.NextButton).clicked.connect(
            self._execute_next_button_action
        )
        self._wizard.button(QWizard.WizardButton.FinishButton).clicked.connect(
            self._execute_finish_button_action
        )

    def _execute_next_button_action(self) -> None:
        page = self._wizard.currentPage()

        if page is self._metadata_view_controller.get_widget():
            self._file_view_controller.open_dataset()
        elif page is self._patterns_view_controller.get_widget():
            self._metadata_view_controller.import_metadata()

    def _execute_finish_button_action(self) -> None:
        self._api.start_assembling_diffraction_patterns()

    def open_dataset(self) -> None:
        self._api.finish_assembling_diffraction_patterns(block=False)
        self._wizard.restart()
        self._file_view_controller.restart()
        self._wizard.show()
