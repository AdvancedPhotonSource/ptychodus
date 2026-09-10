from __future__ import annotations

from PyQt5.QtWidgets import QHBoxLayout, QProgressBar, QPushButton, QWidget


class TaskStatusView(QWidget):
    """Progress bar plus a Stop button for a long-running task.

    Both children stay hidden while nothing is running; TaskStatusController shows
    them for the duration of a task.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.progress_bar = QProgressBar()
        self.stop_button = QPushButton('Stop')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)
