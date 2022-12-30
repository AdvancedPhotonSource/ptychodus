from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog

from ...model.workflow import WorkflowAuthorizationPresenter
from ...view import WorkflowAuthorizationDialog


class WorkflowAuthorizationController:

    def __init__(self, presenter: WorkflowAuthorizationPresenter,
                 dialog: WorkflowAuthorizationDialog) -> None:
        super().__init__()
        self._presenter = presenter
        self._dialog = dialog
        self._timer = QTimer()

    @classmethod
    def createInstance(cls, presenter: WorkflowAuthorizationPresenter,
                       dialog: WorkflowAuthorizationDialog) -> WorkflowAuthorizationController:
        controller = cls(presenter, dialog)

        controller._timer.timeout.connect(controller._startAuthorization)
        controller._timer.start(1000)  # TODO customize
        dialog.finished.connect(controller._finishAuthorization)

        dialog.lineEdit.textChanged.connect(controller._setDialogButtonsEnabled)
        controller._setDialogButtonsEnabled()

        return controller

    def _setDialogButtonsEnabled(self) -> None:
        text = self._dialog.lineEdit.text()
        self._dialog.okButton.setEnabled(len(text) > 0)

    def _startAuthorization(self) -> None:
        if self._presenter.isAuthorized or self._dialog.isVisible():
            return

        authorizeURL = self._presenter.getAuthorizeURL()
        text = f'Input the Globus authorization code from <a href="{authorizeURL}">this link</a>:'

        self._dialog.label.setText(text)
        self._dialog.lineEdit.clear()
        self._dialog.open()

    def _finishAuthorization(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        authCode = self._dialog.lineEdit.text()
        self._presenter.setCodeFromAuthorizeURL(authCode)
