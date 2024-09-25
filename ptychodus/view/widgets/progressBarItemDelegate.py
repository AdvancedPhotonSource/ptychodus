import logging

from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QStyle,
    QStyleOptionProgressBar,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

logger = logging.getLogger(__name__)


class ProgressBarItemDelegate(QStyledItemDelegate):

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        progress = index.data(Qt.ItemDataRole.UserRole)

        if progress is None:
            logger.error(f"Bad data at row={index.row()}, col={index.column()}!")
        elif progress >= 0:
            opt = QStyleOptionProgressBar()
            opt.rect = option.rect
            opt.minimum = 0
            opt.maximum = 100
            opt.progress = progress
            opt.text = f"{progress}%"
            opt.textVisible = True
            QApplication.style().drawControl(QStyle.ControlElement.CE_ProgressBar, opt, painter)
