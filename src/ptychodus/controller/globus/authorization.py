from __future__ import annotations

from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QWidget

from ...model.globus import GlobusAuthorizer
from ...view.globus import GlobusAuthorizationDialog


class GlobusAuthorizationController:
    def __init__(self, authorizer: GlobusAuthorizer, dialog_parent: QWidget) -> None:
        super().__init__()
        self._authorizer = authorizer
        self._dialog = GlobusAuthorizationDialog(dialog_parent)

        self._dialog.finished.connect(self._finish_authorization)
        self._dialog.line_edit.textChanged.connect(self._set_dialog_buttons_enabled)
        self._set_dialog_buttons_enabled()

    def _set_dialog_buttons_enabled(self) -> None:
        ok_button = self._dialog.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setEnabled(len(self._dialog.line_edit.text()) > 0)

    def start_authorization_if_needed(self) -> None:
        if self._authorizer.is_authorized:
            # authorization completed
            return

        if self._dialog.isVisible():
            # authorization in progress
            return

        self._start_authorization()

    def _start_authorization(self) -> None:
        authorize_url = self._authorizer.get_authorize_url()
        text = f'Input the Globus authorization code from <a href="{authorize_url}">this link</a>:'

        self._dialog.label.setText(text)
        self._dialog.line_edit.clear()
        self._dialog.open()

    def _finish_authorization(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            auth_code = self._dialog.line_edit.text()
            self._authorizer.set_code_from_authorize_url(auth_code)
