from typing import Final
import traceback

from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import QMessageBox, QSizePolicy, QTextEdit


class ExceptionDialog(QMessageBox):
    MIN_SIZE: Final[int] = 0
    MAX_SIZE: Final[int] = 16777215

    @classmethod
    def show_exception(cls, actor: str, exception: Exception) -> None:
        dialog = cls()
        dialog.setSizeGripEnabled(True)
        dialog.setWindowTitle('Exception Dialog')
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setText(f'{actor} raised a {exception.__class__.__name__}!')
        dialog.setInformativeText(str(exception))
        # Format the exception's own traceback rather than the one being handled:
        # errors captured on a background thread are replayed from an observer
        # callback, where format_exc() would yield the string 'NoneType: None'.
        dialog.setDetailedText(
            ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        )
        _ = dialog.exec()

    def event(self, e: QEvent) -> bool:
        result = super().event(e)

        if e.type() in (QEvent.Type.LayoutRequest, QEvent.Type.Resize):
            self.setMinimumHeight(self.MIN_SIZE)
            self.setMaximumHeight(self.MAX_SIZE)
            self.setMinimumWidth(self.MIN_SIZE)
            self.setMaximumWidth(self.MAX_SIZE)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            text_edit = self.findChild(QTextEdit)

            if text_edit is not None:
                # make the detailed text expandable
                text_edit.setMinimumHeight(self.MIN_SIZE)
                text_edit.setMaximumHeight(self.MAX_SIZE)
                text_edit.setMinimumWidth(self.MIN_SIZE)
                text_edit.setMaximumWidth(self.MAX_SIZE)
                text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        return result
