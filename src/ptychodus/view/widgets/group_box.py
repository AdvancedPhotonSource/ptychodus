from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMenu,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class GroupBoxWithPresets(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title_label = QLabel(title)

        self.presets_menu = QMenu()
        self._presets_button = QToolButton()
        self._presets_button.setText('Presets  ')
        self._presets_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._presets_button.setMenu(self.presets_menu)
        self._presets_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        self.contents = QWidget()
        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.contents)

        self._frame = QFrame()
        self._frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._frame.setFrameShadow(QFrame.Shadow.Plain)
        self._frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self._frame.setLayout(frame_layout)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._title_label, 0, 0)
        layout.addWidget(self._presets_button, 0, 1, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._frame, 1, 0, 1, 2)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)


class BottomTitledGroupBox(QGroupBox):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: bottom center;
            }""")
