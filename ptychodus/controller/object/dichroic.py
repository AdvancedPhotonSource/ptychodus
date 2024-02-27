from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import DichroicAnalyzer
from ...view.object import DichroicDialog
from .listModel import ObjectListModel


class DichroicViewController:

    def __init__(self, analyzer: DichroicAnalyzer, listModel: ObjectListModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._analyzer = analyzer
        self._dialog = DichroicDialog.createInstance(statusBar, parent)
        self._dialog.lcircNameComboBox.setModel(listModel)
        self._dialog.lcircNameComboBox.textActivated.connect(self._doStuff)
        self._dialog.rcircNameComboBox.setModel(listModel)
        self._dialog.rcircNameComboBox.textActivated.connect(self._doStuff)

    def analyze(self, lcircItemIndex: int, rcircItemIndex: int) -> None:
        self._dialog.lcircNameComboBox.setCurrentIndex(lcircItemIndex)
        self._dialog.rcircNameComboBox.setCurrentIndex(rcircItemIndex)
        self._doStuff()
        self._dialog.open()

    def _doStuff(self) -> None:
        pass  # FIXME
