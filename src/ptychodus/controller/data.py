from collections.abc import Sequence
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog, QWidget

from ptychodus.api.observer import Observable


class FileDialogFactory(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._open_working_directory = Path.cwd()
        self._save_working_directory = Path.cwd()

    def get_open_working_directory(self) -> Path:
        return self._open_working_directory

    def set_open_working_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            directory = directory.parent

        directory = directory.resolve()

        if self._open_working_directory != directory:
            self._open_working_directory = directory
            self.notify_observers()

    def get_open_file_path(
        self,
        parent: QWidget,
        caption: str,
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
    ) -> tuple[Path | None, str]:
        file_path = None

        dialog = QFileDialog(parent, caption, str(self.get_open_working_directory()))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if name_filters is not None:
            dialog.setNameFilters(name_filters)

        if mime_type_filters is not None:
            dialog.setMimeTypeFilters(mime_type_filters)

        if selected_name_filter is not None:
            dialog.selectNameFilter(selected_name_filter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            file_name_list = dialog.selectedFiles()
            file_name = file_name_list[0]

            if file_name:
                file_path = Path(file_name)
                self.set_open_working_directory(file_path.parent)

        return file_path, dialog.selectedNameFilter()

    def get_save_file_path(
        self,
        parent: QWidget,
        caption: str,
        name_filters: Sequence[str] | None = None,
        mime_type_filters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
    ) -> tuple[Path | None, str]:
        file_path = None

        dialog = QFileDialog(parent, caption, str(self._save_working_directory))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)

        if name_filters is not None:
            dialog.setNameFilters(name_filters)

        if mime_type_filters is not None:
            dialog.setMimeTypeFilters(mime_type_filters)

        if selected_name_filter is not None:
            dialog.selectNameFilter(selected_name_filter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            file_name_list = dialog.selectedFiles()
            file_name = file_name_list[0]

            if file_name:
                file_path = Path(file_name)
                self._save_working_directory = file_path.parent

        return file_path, dialog.selectedNameFilter()

    def get_existing_directory_path(
        self, parent: QWidget, caption: str, initial_directory: Path | None = None
    ) -> Path | None:
        dir_path = None

        dir_name = QFileDialog.getExistingDirectory(
            parent,
            caption,
            str(initial_directory or self.get_open_working_directory()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if dir_name:
            dir_path = Path(dir_name)
            self.set_open_working_directory(dir_path)

        return dir_path
