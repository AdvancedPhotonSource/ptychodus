from __future__ import annotations

from PyQt5.QtWidgets import QDialogButtonBox, QListView, QVBoxLayout, QWidget


class SettingsView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.list_view = QListView()
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Save
        )

        layout = QVBoxLayout()
        layout.addWidget(self.list_view)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
