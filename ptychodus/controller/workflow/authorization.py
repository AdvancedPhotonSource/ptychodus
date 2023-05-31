from __future__ import annotations

from PyQt5.QtWidgets import QDialog

from ...model.workflow import WorkflowAuthorizationPresenter
from ...view.workflow import WorkflowAuthorizationDialog


class WorkflowAuthorizationController:

    def __init__(self, presenter: WorkflowAuthorizationPresenter,
                 dialog: WorkflowAuthorizationDialog) -> None:
        super().__init__()
        self._presenter = presenter
        self._dialog = dialog

    @classmethod
    def createInstance(cls, presenter: WorkflowAuthorizationPresenter,
                       dialog: WorkflowAuthorizationDialog) -> WorkflowAuthorizationController:
        controller = cls(presenter, dialog)

        dialog.finished.connect(controller._finishAuthorization)
        dialog.lineEdit.textChanged.connect(controller._setDialogButtonsEnabled)
        controller._setDialogButtonsEnabled()

        return controller

    def _setDialogButtonsEnabled(self) -> None:
        text = self._dialog.lineEdit.text()
        self._dialog.okButton.setEnabled(len(text) > 0)

    def startAuthorizationIfNeeded(self) -> None:
        if not (self._presenter.isAuthorized or self._dialog.isVisible()):
            self._startAuthorization()

    def _startAuthorization(self) -> None:
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
