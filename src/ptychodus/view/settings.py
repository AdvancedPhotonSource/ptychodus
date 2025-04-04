from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QListView, QPushButton, QVBoxLayout, QWidget


class SettingsButtonBox(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.open_button = QPushButton('Open')
        self.save_button = QPushButton('Save')

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> SettingsButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.open_button)
        layout.addWidget(view.save_button)
        view.setLayout(layout)

        return view


class SettingsView(QWidget):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.list_view = QListView()
        self.button_box = SettingsButtonBox.create_instance()

    @classmethod
    def create_instance(cls, parent: QWidget | None = None) -> SettingsView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.list_view)
        layout.addWidget(view.button_box)
        view.setLayout(layout)

        return view
