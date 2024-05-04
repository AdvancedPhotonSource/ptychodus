import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import DichroicAnalyzer, DichroicResult
from ...model.visualization import VisualizationEngine
from ...view.object import DichroicDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController
from .treeModel import ObjectTreeModel

logger = logging.getLogger(__name__)


class DichroicViewController:

    def __init__(self, analyzer: DichroicAnalyzer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._dialog = DichroicDialog.createInstance(parent)
        self._dialog.parametersView.lcircComboBox.setModel(treeModel)
        self._dialog.parametersView.lcircComboBox.currentIndexChanged.connect(self._analyze)
        self._dialog.parametersView.rcircComboBox.setModel(treeModel)
        self._dialog.parametersView.rcircComboBox.currentIndexChanged.connect(self._analyze)
        self._dialog.parametersView.saveButton.clicked.connect(self._saveResult)

        self._differenceVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.differenceWidget, statusBar, fileDialogFactory)
        self._sumVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.sumWidget, statusBar, fileDialogFactory)
        self._ratioVisualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.ratioWidget, statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.parametersView.visualizationParametersView)
        self._result: DichroicResult | None = None

    def _analyze(self) -> None:
        lcircItemIndex = self._dialog.parametersView.lcircComboBox.currentIndex()
        rcircItemIndex = self._dialog.parametersView.rcircComboBox.currentIndex()

        if lcircItemIndex < 0 or rcircItemIndex < 0:
            return

        try:
            result = self._analyzer.analyze(lcircItemIndex, rcircItemIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Dichroic Analysis', err)
            return

        self._result = result
        self._differenceVisualizationWidgetController.setArray(result.polarDifference[0, :, :],
                                                               result.pixelGeometry)
        self._sumVisualizationWidgetController.setArray(result.polarSum[0, :, :],
                                                        result.pixelGeometry)
        # TODO support multi-layer objects
        self._ratioVisualizationWidgetController.setArray(result.polarRatio[0, :, :],
                                                          result.pixelGeometry)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.parametersView.lcircComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.parametersView.rcircComboBox.setCurrentIndex(rcircItemIndex)
        self._analyze()
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
