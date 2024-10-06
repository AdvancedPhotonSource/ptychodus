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


class ProductEditorDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tableView = QTableView()
        self.textEdit = QPlainTextEdit()
        self.buttonBox = QDialogButtonBox()

        commentsLayout = QVBoxLayout()
        commentsLayout.setContentsMargins(0, 0, 0, 0)
        commentsLayout.addWidget(self.textEdit)

        commentsBox = QGroupBox('Comments')
        commentsBox.setLayout(commentsLayout)

        self.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.clicked.connect(self._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.tableView)
        layout.addWidget(commentsBox)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)

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
