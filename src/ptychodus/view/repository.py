from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class RepositoryButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.load_button = QPushButton('Load')
        self.load_menu = QMenu()
        self.save_button = QPushButton('Save')
        self.save_menu = QMenu()
        self.edit_button = QPushButton('Edit')
        self.analyze_button = QPushButton('Analyze')
        self.analyze_menu = QMenu()

        self.load_button.setMenu(self.load_menu)
        self.save_button.setMenu(self.save_menu)
        self.analyze_button.setMenu(self.analyze_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.analyze_button)
        self.setLayout(layout)


class RepositoryItemCopierDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_combo_box = QComboBox()
        self.destination_combo_box = QComboBox()
        self.button_box = QDialogButtonBox()

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.clicked.connect(self._handle_button_box_clicked)

        layout = QFormLayout()
        layout.addRow('From:', self.source_combo_box)
        layout.addRow('To:', self.destination_combo_box)
        layout.addRow(self.button_box)
        self.setLayout(layout)

    def _handle_button_box_clicked(self, button: QAbstractButton) -> None:
        if self.button_box.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class RepositoryTableView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()
        self.button_box = RepositoryButtonBox()
        self.copier_dialog = RepositoryItemCopierDialog()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class RepositoryTreeView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree_view = QTreeView()
        self.button_box = RepositoryButtonBox()
        self.copier_dialog = RepositoryItemCopierDialog()

        header = self.tree_view.header()

        if header is not None:
            header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
