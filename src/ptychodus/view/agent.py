from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QListView,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QPlainTextEdit,
    QWidget,
)


class AgentView(QWidget):
    pass


class AgentInputView(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.text_edit = QPlainTextEdit()

        send_button_size_policy = QSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.send_button = QPushButton(QIcon(':/icons/send'), 'Send')
        self.send_button.setSizePolicy(send_button_size_policy)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.send_button)
        self.setLayout(layout)


class AgentChatView(QSplitter):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Vertical, parent)
        self.message_list_view = QListView()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.message_list_view)
        self.input_view = AgentInputView()

        self.addWidget(self.scroll_area)
        self.addWidget(self.input_view)

        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 0)
