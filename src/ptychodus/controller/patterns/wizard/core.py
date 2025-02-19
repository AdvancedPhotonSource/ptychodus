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
            self._api, file_dialog_factory
        )
        self._metadata_view_controller = OpenDatasetWizardMetadataViewController(metadata_presenter)
        self._patterns_view_controller = OpenDatasetWizardPatternsViewController(
            settings, sizer, file_dialog_factory
        )

        self._wizard = QWizard()
        self._wizard.setWindowTitle('Open Dataset')
        self._wizard.addPage(self._file_view_controller.getWidget())
        self._wizard.addPage(self._metadata_view_controller.getWidget())
        self._wizard.addPage(self._patterns_view_controller.getWidget())

        self._wizard.button(QWizard.WizardButton.NextButton).clicked.connect(
            self._executeNextButtonAction
        )
        self._wizard.button(QWizard.WizardButton.FinishButton).clicked.connect(
            self._executeFinishButtonAction
        )

    def _executeNextButtonAction(self) -> None:
        page = self._wizard.currentPage()

        if page is self._metadata_view_controller.getWidget():
            self._file_view_controller.openDataset()
        elif page is self._patterns_view_controller.getWidget():
            self._metadata_view_controller.importMetadata()

    def _executeFinishButtonAction(self) -> None:
        self._api.startAssemblingDiffractionPatterns()

    def openDataset(self) -> None:
        self._api.finishAssemblingDiffractionPatterns(block=False)
        self._wizard.restart()
        self._wizard.show()
