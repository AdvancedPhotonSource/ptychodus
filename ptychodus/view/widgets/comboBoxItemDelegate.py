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
        self._paintComboBox = False

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if self._paintComboBox and index.flags() & Qt.ItemFlag.ItemIsEditable:
            opt = QStyleOptionComboBox()
            opt.rect = option.rect
            opt.currentText = index.data(Qt.DisplayRole)
            QApplication.style().drawComplexControl(
                QStyle.ComplexControl.CC_ComboBox, opt, painter
            )
            QApplication.style().drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt, painter)
        else:
            super().paint(painter, option, index)

    def createEditor(
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        comboBox = QComboBox(parent)
        comboBox.activated.connect(self._commitDataAndCloseEditor)
        comboBox.setModel(self._model)
        return comboBox

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:
        if isinstance(editor, QComboBox):
            currentText = str(index.data(Qt.EditRole))
            comboBoxIndex = editor.findText(currentText)

            if comboBoxIndex >= 0:
                editor.setCurrentIndex(comboBoxIndex)

            editor.showPopup()
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex) -> None:
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.EditRole)
        else:
            super().setModelData(editor, model, index)

    def updateEditorGeometry(
        self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> None:
        editor.setGeometry(option.rect)

    def _commitDataAndCloseEditor(self) -> None:
        editor = self.sender()

        if isinstance(editor, QComboBox):
            self.commitData.emit(editor)
            self.closeEditor.emit(editor)
        else:
            logger.warning("Failed to commit data and close editor! Unexpected editor.")
