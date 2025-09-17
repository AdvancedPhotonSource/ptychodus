import logging

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QWidget,
)

logger = logging.getLogger(__name__)


class ComboBoxItemDelegate(QStyledItemDelegate):
    # https://wiki.qt.io/Combo_Boxes_in_Item_Views

    def __init__(self, model: QAbstractItemModel, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._paint_combo_box = False

    def paint(
        self, painter: QPainter | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if painter is None:
            return

        if self._paint_combo_box and index.flags() & Qt.ItemFlag.ItemIsEditable:
            opt = QStyleOptionComboBox()
            opt.rect = option.rect
            opt.currentText = index.data(Qt.ItemDataRole.DisplayRole)
            style = QApplication.style()

            if style is not None:
                style.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt, painter)
                style.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt, painter)
        else:
            super().paint(painter, option, index)

    def createEditor(  # noqa: N802
        self, parent: QWidget | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        combo_box = QComboBox(parent)
        combo_box.activated.connect(self._commit_data_and_close_editor)
        combo_box.setModel(self._model)
        return combo_box

    def setEditorData(self, editor: QWidget | None, index: QModelIndex) -> None:  # noqa: N802
        if editor is None:
            return

        if isinstance(editor, QComboBox):
            current_text = str(index.data(Qt.ItemDataRole.EditRole))
            combo_box_index = editor.findText(current_text)

            if combo_box_index >= 0:
                editor.setCurrentIndex(combo_box_index)

            editor.showPopup()
        else:
            super().setEditorData(editor, index)

    def setModelData(  # noqa: N802
        self, editor: QWidget | None, model: QAbstractItemModel | None, index: QModelIndex
    ) -> None:
        if editor is None or model is None:
            return

        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)

    def updateEditorGeometry(  # noqa: N802
        self, editor: QWidget | None, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        if editor is None:
            return

        editor.setGeometry(option.rect)

    def _commit_data_and_close_editor(self) -> None:
        editor = self.sender()

        if isinstance(editor, QComboBox):
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)
        else:
            logger.warning('Failed to commit data and close editor! Unexpected editor.')
