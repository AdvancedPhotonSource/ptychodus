from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QGroupBox, QHeaderView, QHBoxLayout, QMenu, QPushButton, QTableView,
                             QVBoxLayout, QWidget)


class RepositoryButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> RepositoryButtonBox:
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


class RepositoryView(QGroupBox):

    def __init__(self, title: str, parent: Optional[QWidget]) -> None:
        super().__init__(title, parent)
        self.tableView = QTableView()
        self.buttonBox = RepositoryButtonBox.createInstance()

    @classmethod
    def createInstance(cls, title: str, parent: Optional[QWidget] = None) -> RepositoryView:
        view = cls(title, parent)

        view.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view
