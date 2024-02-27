from __future__ import annotations

from PyQt5.QtWidgets import (QAbstractButton, QDialog, QDialogButtonBox, QStatusBar, QVBoxLayout,
                             QWidget)

from .image import ImageView


class ProbePropagationDialog(QDialog):

    def __init__(self, statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.imageView = ImageView.createInstance(statusBar)
        self.buttonBox = QDialogButtonBox()

    @classmethod
    def createInstance(cls,
                       statusBar: QStatusBar,
                       parent: QWidget | None = None) -> ProbePropagationDialog:
        view = cls(statusBar, parent)
        view.setWindowTitle('Probe Propagation')

        view.buttonBox.addButton(QDialogButtonBox.StandardButton.Ok)
        view.buttonBox.clicked.connect(view._handleButtonBoxClicked)

        layout = QVBoxLayout()
        layout.addWidget(view.imageView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view

    def _handleButtonBoxClicked(self, button: QAbstractButton) -> None:
        if self.buttonBox.buttonRole(button) == QDialogButtonBox.ButtonRole.AcceptRole:
            self.accept()
        else:
            self.reject()
