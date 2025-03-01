from typing import Final
import traceback

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QTextEdit


class ExceptionDialog(QMessageBox):
    MIN_SIZE: Final[int] = 0
    MAX_SIZE: Final[int] = 16777215

    @classmethod
    def showException(cls, actor: str, exception: Exception) -> None:
        dialog = cls()
        dialog.setSizeGripEnabled(True)
        dialog.setWindowTitle('Exception Dialog')
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setText(f'{actor} raised a {exception.__class__.__name__}!')
        dialog.setInformativeText(str(exception))
        dialog.setDetailedText(traceback.format_exc())
        _ = dialog.exec()

    def event(self, event: QEvent) -> bool:
        result = super().event(event)

        if event.type() == QEvent.LayoutRequest or event.type() == QEvent.Resize:
            self.setMinimumHeight(self.MIN_SIZE)
            self.setMaximumHeight(self.MAX_SIZE)
            self.setMinimumWidth(self.MIN_SIZE)
            self.setMaximumWidth(self.MAX_SIZE)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            textEdit = self.findChild(QTextEdit)

            if textEdit is not None:
                # make the detailed text expandable
                textEdit.setMinimumHeight(self.MIN_SIZE)
                textEdit.setMaximumHeight(self.MAX_SIZE)
                textEdit.setMinimumWidth(self.MIN_SIZE)
                textEdit.setMaximumWidth(self.MAX_SIZE)
                textEdit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        return result
