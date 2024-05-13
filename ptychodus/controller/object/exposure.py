import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ExposureAnalyzer, ExposureMap
from ...model.visualization import VisualizationEngine
from ...view.object import ExposureDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class ExposureViewController:

    def __init__(self, analyzer: ExposureAnalyzer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, statusBar: QStatusBar,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._dialog = ExposureDialog.createInstance(parent)
        self._dialog.saveButton.clicked.connect(self._saveResult)

        self._visualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.visualizationWidget, statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView)
        self._result: ExposureMap | None = None

    def analyze(self, itemIndex: int) -> None:
        try:
            result = self._analyzer.analyze(itemIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Exposure Analysis', err)
            return

        self._result = result
        self._dialog.open()

    def _saveResult(self) -> None:
        if self._result is None:
            logger.debug('No result to save!')
            return

        title = 'Save Result'
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            title,
            nameFilters=self._analyzer.getSaveFileFilterList(),
            selectedNameFilter=self._analyzer.getSaveFileFilter())

        if filePath:
            try:
                self._analyzer.saveResult(filePath, self._result)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)
