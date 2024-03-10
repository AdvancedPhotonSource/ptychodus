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
        self._dialog = DichroicDialog.createInstance(statusBar, parent)
        self._dialog.parametersView.lcircComboBox.setModel(treeModel)
        self._dialog.parametersView.lcircComboBox.textActivated.connect(self._doStuff)
        self._dialog.parametersView.rcircComboBox.setModel(treeModel)
        self._dialog.parametersView.rcircComboBox.textActivated.connect(self._doStuff)
        self._ratioImageController = ImageController.createInstance(
            imagePresenter, self._dialog.ratioImageView.imageView, fileDialogFactory)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.parametersView.lcircComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.parametersView.rcircComboBox.setCurrentIndex(rcircItemIndex)
        self._doStuff()
        self._dialog.open()

    def _doStuff(self) -> None:
        pass  # FIXME
