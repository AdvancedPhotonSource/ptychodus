from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QListView, QPushButton, QVBoxLayout, QWidget


class SettingsButtonBox(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.open_button = QPushButton('Open')
        self.save_button = QPushButton('Save')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.open_button)
        layout.addWidget(self.save_button)
        self.setLayout(layout)


class SettingsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list_view = QListView()
        self.button_box = SettingsButtonBox()

        layout = QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
