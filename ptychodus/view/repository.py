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
        self.loadButton = QPushButton('Load')
        self.loadMenu = QMenu()
        self.saveButton = QPushButton('Save')
        self.saveMenu = QMenu()
        self.editButton = QPushButton('Edit')
        self.analyzeButton = QPushButton('Analyze')
        self.analyzeMenu = QMenu()

        self.loadButton.setMenu(self.loadMenu)
        self.saveButton.setMenu(self.saveMenu)
        self.analyzeButton.setMenu(self.analyzeMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.loadButton)
        layout.addWidget(self.saveButton)
        layout.addWidget(self.editButton)
        layout.addWidget(self.analyzeButton)
        self.setLayout(layout)


class RepositoryItemCopierDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.sourceComboBox = QComboBox()
        self.destinationComboBox = QComboBox()
        self.buttonBox = QDialogButtonBox()

        self.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.clicked.connect(self._handleButtonBoxClicked)

        layout = QFormLayout()
        layout.addRow('From:', self.sourceComboBox)
        layout.addRow('To:', self.destinationComboBox)
        layout.addRow(self.buttonBox)
        self.setLayout(layout)

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class RepositoryTableView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.buttonBox = RepositoryButtonBox()
        self.copierDialog = RepositoryItemCopierDialog()

        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)


class RepositoryTreeView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.treeView = QTreeView()
        self.buttonBox = RepositoryButtonBox()
        self.copierDialog = RepositoryItemCopierDialog()

        self.treeView.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.treeView)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
