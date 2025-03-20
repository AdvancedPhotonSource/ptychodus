from __future__ import annotations

from PyQt5.QtWidgets import (
    QAbstractButton,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class ProductEditorPropertiesView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Properties')
        self.tableView = QTableView()

        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        self.setLayout(layout)


class ProductEditorActionsView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Actions')
        self.estimateProbePhotonCountButton = QPushButton('Estimate Probe Photon Count')

        layout = QVBoxLayout()
        layout.addWidget(self.estimateProbePhotonCountButton)
        layout.addStretch()
        self.setLayout(layout)


class ProductEditorCommentsView(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Comments')
        self.textEdit = QPlainTextEdit()

        layout = QVBoxLayout()
        layout.addWidget(self.textEdit)
        self.setLayout(layout)


class ProductEditorDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.propertiesView = ProductEditorPropertiesView()
        self.actionsView = ProductEditorActionsView()
        self.commentsView = ProductEditorCommentsView()
        self.buttonBox = QDialogButtonBox()

        self.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.clicked.connect(self._handleButtonBoxClicked)

        topLayout = QHBoxLayout()
        topLayout.addWidget(self.propertiesView)
        topLayout.addWidget(self.actionsView)

        layout = QVBoxLayout()
        layout.addLayout(topLayout)
        layout.addWidget(self.commentsView)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

    @property
    def tableView(self) -> QTableView:
        return self.propertiesView.tableView

    @property
    def textEdit(self) -> QPlainTextEdit:
        return self.commentsView.textEdit

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()


class ProductButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.insertMenu = QMenu()
        self.insertButton = QPushButton('Insert')
        self.saveMenu = QMenu()
        self.saveButton = QPushButton('Save')
        self.editButton = QPushButton('Edit')
        self.removeButton = QPushButton('Remove')

        self.insertButton.setMenu(self.insertMenu)
        self.saveButton.setMenu(self.saveMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.insertButton)
        layout.addWidget(self.saveButton)
        layout.addWidget(self.editButton)
        layout.addWidget(self.removeButton)
        self.setLayout(layout)


class ProductView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.infoLabel = QLabel()
        self.buttonBox = ProductButtonBox()

        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        layout.addWidget(self.infoLabel)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)
