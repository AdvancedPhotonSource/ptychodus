from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QListView, QPushButton, QVBoxLayout, QWidget


class SettingsButtonBox(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> SettingsButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.openButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class SettingsView(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.listView = QListView()
        self.buttonBox = SettingsButtonBox.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> SettingsView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.listView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view
