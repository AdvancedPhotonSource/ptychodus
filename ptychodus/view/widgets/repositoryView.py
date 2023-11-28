from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QHBoxLayout, QMenu, QPushButton, QTableView, QTreeView, QVBoxLayout,
                             QWidget)


class RepositoryButtonBox(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.editButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class RepositoryTableView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.buttonBox = RepositoryButtonBox.createInstance()

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

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> RepositoryTreeView:
        view = cls(parent)
        view.treeView.header().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view
