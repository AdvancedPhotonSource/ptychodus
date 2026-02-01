from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...model.globus import GlobusAuthorizer


class GlobusAuthorizationDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.line_edit = QLineEdit()
        self.button_box = QDialogButtonBox()

        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.label.setOpenExternalLinks(True)

        self.button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        self.button_box.accepted.connect(self.accept)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.rejected.connect(self.reject)

        self.setWindowTitle('Authorize Workflow')

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


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

    def authorize_as_needed(self) -> None:
        if self._authorizer.needs_authorize_code and not self._dialog.isVisible():
            self._start_authorization()

    def _start_authorization(self) -> None:
        authorize_url = self._authorizer.get_authorize_url()
        text = f'Input the Globus authorization code from <a href="{authorize_url}">this link</a>:'

        self._dialog.label.setText(text)
        self._dialog.line_edit.clear()
        self._dialog.setModal(True)
        self._dialog.open()

    def _finish_authorization(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            auth_code = self._dialog.line_edit.text()
            self._authorizer.set_code_from_authorize_url(auth_code)
        else:
            self._authorizer.cancel_authorization()
