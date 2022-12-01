from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (QApplication, QStyle, QStyleOptionProgressBar, QStyledItemDelegate,
                             QStyleOptionViewItem)


class ProgressBarItemDelegate(QStyledItemDelegate):

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        progress = index.data(Qt.UserRole)
        opt = QStyleOptionProgressBar()
        opt.rect = option.rect
        opt.minimum = 0
        opt.maximum = 100
        opt.progress = progress
        opt.text = f'{progress}%'
        opt.textVisible = True
        QApplication.style().drawControl(QStyle.CE_ProgressBar, opt, painter)
