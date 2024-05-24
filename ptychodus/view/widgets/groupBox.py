from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QFrame, QGridLayout, QGroupBox, QLabel, QMenu, QSizePolicy,
                             QToolButton, QVBoxLayout, QWidget)


class GroupBoxWithPresets(QWidget):

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._titleLabel = QLabel(title)

        self.presetsMenu = QMenu()
        self._presetsButton = QToolButton()
        self._presetsButton.setText('Presets  ')
        self._presetsButton.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self._presetsButton.setMenu(self.presetsMenu)
        self._presetsButton.setPopupMode(QToolButton.InstantPopup)

        self.contents = QWidget()
        frameLayout = QVBoxLayout()
        frameLayout.addWidget(self.contents)

        self._frame = QFrame()
        self._frame.setFrameShape(QFrame.StyledPanel)
        self._frame.setFrameShadow(QFrame.Plain)
        self._frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self._frame.setLayout(frameLayout)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._titleLabel, 0, 0)
        layout.addWidget(self._presetsButton, 0, 1, Qt.AlignLeft)
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
