import traceback

from PyQt5.QtWidgets import QMessageBox


class ExceptionDialog(QMessageBox):

    @classmethod
    def showException(cls, actor: str, exception: Exception) -> None:
        dialog = cls()
        dialog.setWindowTitle('Exception Dialog')
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setText(f'{actor} raised a {exception.__class__.__name__}!')
        dialog.setInformativeText(str(exception))
        dialog.setDetailedText(traceback.format_exc())
        _ = dialog.open()
