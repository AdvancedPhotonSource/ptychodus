import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import DichroicAnalyzer
from ...model.visualization import VisualizationEngine
from ...view.object import DichroicDialog
from ..data import FileDialogFactory
from .treeModel import ObjectTreeModel

logger = logging.getLogger(__name__)


class DichroicViewController:

    def __init__(self, analyzer: DichroicAnalyzer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._engine = engine
        self._dialog = DichroicDialog.createInstance(statusBar, parent)
        self._dialog.parametersView.lcircComboBox.setModel(treeModel)
        self._dialog.parametersView.lcircComboBox.currentIndexChanged.connect(self._analyze)
        self._dialog.parametersView.rcircComboBox.setModel(treeModel)
        self._dialog.parametersView.rcircComboBox.currentIndexChanged.connect(self._analyze)
        # FIXME self._ratioImageController = ImageController.createInstance(engine, self._dialog.ratioImageView.imageView, fileDialogFactory)

    def _analyze(self) -> None:
        lcircItemIndex = self._dialog.parametersView.lcircComboBox.currentIndex()
        rcircItemIndex = self._dialog.parametersView.rcircComboBox.currentIndex()

        if lcircItemIndex < 0 or rcircItemIndex < 0:
            return

        results = self._analyzer.analyze(lcircItemIndex, rcircItemIndex)
        polarRatio = results.polarRatio[0, :, :]  # TODO support multislice
        # FIXME self._engine.setArray(polarRatio, results.pixelGeometry)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.parametersView.lcircComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.parametersView.rcircComboBox.setCurrentIndex(rcircItemIndex)
        self._analyze()
        self._dialog.open()
