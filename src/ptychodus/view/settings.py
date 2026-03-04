from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QListView,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)


class SettingsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list_view = QListView()
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Save
        )

        layout = QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class SyncProductToSettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.icon_label = QLabel()
        self.prompt_label = QLabel('Update settings registry with product parameters?')
        self.product_label = QLabel('Product:')
        self.product_combo_box = QComboBox()
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        style = self.style()
        icon = style.standardIcon(QStyle.SP_MessageBoxQuestion)
        icon_size = style.pixelMetric(QStyle.PM_MessageBoxIconSize)
        pixmap = icon.pixmap(QSize(icon_size, icon_size))
        self.icon_label.setPixmap(pixmap)

        indent_spacer = QSpacerItem(7, QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QGridLayout()
        layout.addWidget(self.icon_label, 0, 0, 2, 1, Qt.AlignTop)
        layout.addItem(indent_spacer, 0, 1, 2, 1)
        layout.addWidget(self.prompt_label, 0, 2, 1, 2)
        layout.addWidget(self.product_label, 1, 2, 1, 1)
        layout.addWidget(self.product_combo_box, 1, 3, 1, 1)
        layout.addWidget(self.button_box, 2, 0, 1, 4)
        self.setLayout(layout)

        self.setWindowTitle('Sync Data Product To Settings')
