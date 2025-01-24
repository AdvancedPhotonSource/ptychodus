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
        self.textEdit = QPlainTextEdit()

        sendButtonSizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.sendButton = QPushButton(QIcon(':/icons/send'), 'Send')
        self.sendButton.setSizePolicy(sendButtonSizePolicy)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.textEdit)
        layout.addWidget(self.sendButton)
        self.setLayout(layout)


class AgentChatView(QSplitter):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Vertical, parent)
        self.messageListView = QListView()
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.messageListView)
        self.inputView = AgentInputView()

        self.addWidget(self.scrollArea)
        self.addWidget(self.inputView)

        self.setStretchFactor(0, 1)
        self.setStretchFactor(1, 0)
