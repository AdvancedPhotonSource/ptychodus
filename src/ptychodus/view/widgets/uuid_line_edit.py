from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QLineEdit, QWidget


class UUIDLineEdit(QLineEdit):
    @staticmethod
    def _create_validator() -> QRegularExpressionValidator:
        hexre = '[0-9A-Fa-f]'
        uuidre = f'{hexre}{{8}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{4}}-{hexre}{{12}}'
        return QRegularExpressionValidator(QRegularExpression(uuidre))

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setValidator(UUIDLineEdit._create_validator())
