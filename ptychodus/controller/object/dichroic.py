import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import DichroicAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import DichroicDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationController
from .treeModel import ObjectTreeModel

logger = logging.getLogger(__name__)


class DichroicViewController:

    def __init__(self, analyzer: DichroicAnalyzer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine
        self._dialog = DichroicDialog.createInstance(parent)
        self._dialog.parametersView.lcircComboBox.setModel(treeModel)
        self._dialog.parametersView.lcircComboBox.currentIndexChanged.connect(self._analyze)
        self._dialog.parametersView.rcircComboBox.setModel(treeModel)
        self._dialog.parametersView.rcircComboBox.currentIndexChanged.connect(self._analyze)
        self._differenceVisualizationController = VisualizationController.createInstance(
            engine, self._dialog.differenceWidget.visualizationView, statusBar, fileDialogFactory)
        self._sumVisualizationController = VisualizationController.createInstance(
            engine, self._dialog.sumWidget.visualizationView, statusBar, fileDialogFactory)
        self._ratioVisualizationController = VisualizationController.createInstance(
            engine, self._dialog.ratioWidget.visualizationView, statusBar, fileDialogFactory)

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

        # TODO support multi-layer objects
        self._differenceVisualizationController.setArray(result.polarDifference[0, :, :],
                                                         result.pixelGeometry)
        self._sumVisualizationController.setArray(result.polarSum[0, :, :], result.pixelGeometry)
        self._ratioVisualizationController.setArray(result.polarRatio[0, :, :],
                                                    result.pixelGeometry)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.parametersView.lcircComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.parametersView.rcircComboBox.setCurrentIndex(rcircItemIndex)
        self._analyze()
        self._dialog.open()
