from __future__ import annotations

from PyQt5.QtWidgets import QAbstractButton, QDialog, QDialogButtonBox, QVBoxLayout, QWidget


class ProbePropagationDialog(QDialog):  # FIXME use this

    def __init__(self, buttonBox: QDialogButtonBox, parent: QWidget | None) -> None:
        super().__init__(parent)
        self._buttonBox = buttonBox

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> ProbePropagationDialog:
        buttonBox = QDialogButtonBox()
        view = cls(buttonBox, parent)
        view.setWindowTitle('Probe Propagation')

        buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self._buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
