from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import DichroicAnalyzer
from ...model.image import ImagePresenter
from ...view.object import DichroicDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .treeModel import ObjectTreeModel


class DichroicViewController:

    def __init__(self, analyzer: DichroicAnalyzer, imagePresenter: ImagePresenter,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._imagePresenter = imagePresenter
        self._dialog = DichroicDialog.createInstance(statusBar, parent)
        self._dialog.parametersView.lcircComboBox.setModel(treeModel)
        self._dialog.parametersView.lcircComboBox.currentIndexChanged.connect(self._analyze)
        self._dialog.parametersView.rcircComboBox.setModel(treeModel)
        self._dialog.parametersView.rcircComboBox.currentIndexChanged.connect(self._analyze)
        self._ratioImageController = ImageController.createInstance(
            imagePresenter, self._dialog.ratioImageView.imageView, fileDialogFactory)

    def _analyze(self) -> None:
        lcircItemIndex = self._dialog.parametersView.lcircComboBox.currentIndex()
        rcircItemIndex = self._dialog.parametersView.rcircComboBox.currentIndex()
        results = self._analyzer.analyze(lcircItemIndex, rcircItemIndex)
        polarRatio = results.polarRatio[0, :, :]  # TODO support multislice
        self._imagePresenter.setArray(polarRatio, results.pixelGeometry)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.parametersView.lcircComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.parametersView.rcircComboBox.setCurrentIndex(rcircItemIndex)
        self._analyze()
        self._dialog.open()
