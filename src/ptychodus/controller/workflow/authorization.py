from __future__ import annotations

from PyQt5.QtWidgets import QDialog

from ...model.workflow import WorkflowAuthorizationPresenter
from ...view.workflow import WorkflowAuthorizationDialog


class WorkflowAuthorizationController:
    def __init__(
        self,
        presenter: WorkflowAuthorizationPresenter,
        dialog: WorkflowAuthorizationDialog,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._dialog = dialog

        dialog.finished.connect(self._finish_authorization)
        dialog.line_edit.textChanged.connect(self._set_dialog_buttons_enabled)
        self._set_dialog_buttons_enabled()

    def _set_dialog_buttons_enabled(self) -> None:
        text = self._dialog.line_edit.text()
        ok_button = self._dialog.ok_button

        if ok_button is not None:
            ok_button.setEnabled(len(text) > 0)

    def start_authorization_if_needed(self) -> None:
        if not (self._presenter.is_authorized or self._dialog.isVisible()):
            self._start_authorization()

    def _start_authorization(self) -> None:
        authorize_url = self._presenter.get_authorize_url()
        text = f'Input the Globus authorization code from <a href="{authorize_url}">this link</a>:'

        self._dialog.label.setText(text)
        self._dialog.line_edit.clear()
        self._dialog.open()

    def _finish_authorization(self, result: int) -> None:
        if result != QDialog.DialogCode.Accepted:
            return

        auth_code = self._dialog.line_edit.text()
        self._presenter.set_code_from_authorize_url(auth_code)
