from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractButton, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QHBoxLayout, QMenu, QPushButton, QTableView, QTreeView, QVBoxLayout,
                             QWidget)


class RepositoryButtonBox(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.loadButton = QPushButton('Load')
        self.loadMenu = QMenu()
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.analyzeButton = QPushButton('Analyze')
        self.analyzeMenu = QMenu()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryButtonBox:
        view = cls(parent)

        view.loadButton.setMenu(view.loadMenu)
        view.analyzeButton.setMenu(view.analyzeMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.loadButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.editButton)
        layout.addWidget(view.analyzeButton)
        view.setLayout(layout)

        return view


class RepositoryItemCopierDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.sourceComboBox = QComboBox()
        self.destinationComboBox = QComboBox()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryItemCopierDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Cancel)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QFormLayout()
        layout.addRow('From:', view.sourceComboBox)
        layout.addRow('To:', view.destinationComboBox)
        layout.addRow(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class RepositoryTableView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.buttonBox = RepositoryButtonBox.createInstance()
        self.copierDialog = RepositoryItemCopierDialog.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryTableView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class RepositoryTreeView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.treeView = QTreeView()
        self.buttonBox = RepositoryButtonBox.createInstance()
        self.copierDialog = RepositoryItemCopierDialog.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryTreeView:
        view = cls(parent)
        view.treeView.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view
