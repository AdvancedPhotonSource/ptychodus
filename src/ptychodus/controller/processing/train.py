"""Collecting the two directories a local training run needs.

Training reads from one directory and writes to another, and has since the
interface was written; only the GUI used to collapse them, passing one chosen
path as both and leaving every backend's checkpoints beside the data it learned
from. This asks for the two separately.

The pair is remembered for the session rather than persisted. They are
arguments to one run, not configuration: a settings key for either would stand
beside the directories batch mode is given on the command line, with nothing to
say which of the two a run should believe.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtWidgets import QDialogButtonBox, QLineEdit, QWidget

from ...view.processing import TrainDialog
from ..data import FileDialogFactory

__all__ = ['TrainDialogController']


class TrainDialogController:
    """Runs the train dialog and reports the directories it collected."""

    def __init__(self, parent: QWidget, file_dialog_factory: FileDialogFactory) -> None:
        self._dialog = TrainDialog(parent)
        self._file_dialog_factory = file_dialog_factory

        self._dialog.data_browse_button.clicked.connect(
            lambda: self._browse(self._dialog.data_line_edit, 'Choose Training Data Directory')
        )
        self._dialog.output_browse_button.clicked.connect(
            lambda: self._browse(self._dialog.output_line_edit, 'Choose Model Output Directory')
        )
        self._dialog.data_line_edit.textChanged.connect(self._sync_accept_to_view)
        self._dialog.output_line_edit.textChanged.connect(self._sync_accept_to_view)
        self._sync_accept_to_view()

    def _browse(self, line_edit: QLineEdit, caption: str) -> None:
        text = line_edit.text().strip()
        initial = Path(text) if text else None
        dir_path = self._file_dialog_factory.get_existing_directory_path(
            self._dialog, caption, initial_directory=initial
        )

        if dir_path is not None:
            line_edit.setText(str(dir_path))

    def _sync_accept_to_view(self) -> None:
        """A run needs both directories, so offer Ok only once both are named."""
        complete = bool(self._dialog.data_line_edit.text().strip()) and bool(
            self._dialog.output_line_edit.text().strip()
        )
        button = self._dialog.button_box.button(QDialogButtonBox.StandardButton.Ok)

        if button is not None:
            button.setEnabled(complete)

    def get_directories(self) -> tuple[Path, Path] | None:
        """Run the dialog, returning the data and output directories, or None.

        The fields keep whatever the last accepted run used, so repeating one
        is a single click.
        """
        if self._dialog.exec() != TrainDialog.DialogCode.Accepted:
            return None

        return (
            Path(self._dialog.data_line_edit.text().strip()),
            Path(self._dialog.output_line_edit.text().strip()),
        )
