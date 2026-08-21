from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


class InsertSaveEditRemoveButtonBox(QWidget):
    """Insert / Save / Edit / Remove strip shared by the repository-style panels.

    Insert and Save carry menus that each panel's controller populates; the menu
    *contents* are per-domain, the widget is not.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.insert_menu = QMenu()
        self.insert_button = QPushButton('Insert')
        self.save_menu = QMenu()
        self.save_button = QPushButton('Save')
        self.edit_button = QPushButton('Edit')
        self.remove_button = QPushButton('Remove')

        self.insert_button.setMenu(self.insert_menu)
        self.save_button.setMenu(self.save_menu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.insert_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.edit_button)
        layout.addWidget(self.remove_button)
        self.setLayout(layout)


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
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        layout = QFormLayout()
        layout.addRow('From:', self.source_combo_box)
        layout.addRow('To:', self.destination_combo_box)
        layout.addRow(self.button_box)
        self.setLayout(layout)


class RepositoryTableView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table_view = QTableView()
        self.info_label = QLabel()
        self.button_box = RepositoryButtonBox()
        self.copier_dialog = RepositoryItemCopierDialog()

        layout = QVBoxLayout()
        layout.addWidget(self.table_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class RepositoryTreeView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tree_view = QTreeView()
        self.info_label = QLabel()
        self.button_box = RepositoryButtonBox()
        self.copier_dialog = RepositoryItemCopierDialog()

        header = self.tree_view.header()

        if header is not None:
            header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.tree_view)
        layout.addWidget(self.info_label)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
