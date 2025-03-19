from collections.abc import Sequence
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QFileDialog, QWidget

from ptychodus.api.observer import Observable


class FileDialogFactory(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._openWorkingDirectory = Path.cwd()
        self._saveWorkingDirectory = Path.cwd()

    def getOpenWorkingDirectory(self) -> Path:
        return self._openWorkingDirectory

    def setOpenWorkingDirectory(self, directory: Path) -> None:
        if not directory.is_dir():
            directory = directory.parent

        directory = directory.resolve()

        if self._openWorkingDirectory != directory:
            self._openWorkingDirectory = directory
            self.notify_observers()

    def get_open_file_path(
        self,
        parent: QWidget,
        caption: str,
        name_filters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
    ) -> tuple[Path | None, str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self.getOpenWorkingDirectory()))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if name_filters is not None:
            dialog.setNameFilters(name_filters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selected_name_filter is not None:
            dialog.selectNameFilter(selected_name_filter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self.setOpenWorkingDirectory(filePath.parent)

        return filePath, dialog.selectedNameFilter()

    def get_save_file_path(
        self,
        parent: QWidget,
        caption: str,
        name_filters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selected_name_filter: str | None = None,
    ) -> tuple[Path | None, str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self._saveWorkingDirectory))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)

        if name_filters is not None:
            dialog.setNameFilters(name_filters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selected_name_filter is not None:
            dialog.selectNameFilter(selected_name_filter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self._saveWorkingDirectory = filePath.parent

        return filePath, dialog.selectedNameFilter()

    def get_existing_directory_path(
        self, parent: QWidget, caption: str, initial_directory: Path | None = None
    ) -> Path | None:
        dirPath = None

        dirName = QFileDialog.getExistingDirectory(
            parent,
            caption,
            str(initial_directory or self.getOpenWorkingDirectory()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if dirName:
            dirPath = Path(dirName)
            self.setOpenWorkingDirectory(dirPath)

        return dirPath
