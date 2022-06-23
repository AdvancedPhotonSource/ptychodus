from typing import Optional

from PyQt5.QtWidgets import QGroupBox, QWidget


class BottomTitledGroupBox(QGroupBox):

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setStyleSheet("""
            QGroupBox::title {
                subcontrol-origin: padding;
                subcontrol-position: bottom center;
            }""")
