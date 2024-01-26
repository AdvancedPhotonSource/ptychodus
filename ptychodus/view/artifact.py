from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
                             QLabel, QMenu, QPushButton, QTableView, QVBoxLayout, QWidget)


class ArtifactInfoDialog(QDialog):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ArtifactInfoDialog:
        view = cls(parent)

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ArtifactButtonBox(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveButton = QPushButton('Save')
        self.infoButton = QPushButton('Info')
        self.removeButton = QPushButton('Remove')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ArtifactButtonBox:
        view = cls(parent)

        view.insertButton.setMenu(view.insertMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.insertButton)
        layout.addWidget(view.saveButton)
        layout.addWidget(view.infoButton)
        layout.addWidget(view.removeButton)
        view.setLayout(layout)

        return view


class ArtifactRepositoryView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Repository', parent)
        self.tableView = QTableView()
        self.infoLabel = QLabel()
        self.buttonBox = ArtifactButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ArtifactRepositoryView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.tableView)
        layout.addWidget(view.infoLabel)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ArtifactView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.repositoryView = ArtifactRepositoryView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ArtifactView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view
