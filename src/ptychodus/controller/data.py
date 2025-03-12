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

    def getOpenFilePath(
        self,
        parent: QWidget,
        caption: str,
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
    ) -> tuple[Path | None, str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self.getOpenWorkingDirectory()))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if nameFilters is not None:
            dialog.setNameFilters(nameFilters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selectedNameFilter is not None:
            dialog.selectNameFilter(selectedNameFilter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self.setOpenWorkingDirectory(filePath.parent)

        return filePath, dialog.selectedNameFilter()

    def getSaveFilePath(
        self,
        parent: QWidget,
        caption: str,
        nameFilters: Sequence[str] | None = None,
        mimeTypeFilters: Sequence[str] | None = None,
        selectedNameFilter: str | None = None,
    ) -> tuple[Path | None, str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self._saveWorkingDirectory))
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)

        if nameFilters is not None:
            dialog.setNameFilters(nameFilters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selectedNameFilter is not None:
            dialog.selectNameFilter(selectedNameFilter)

        if dialog.exec() == QDialog.DialogCode.Accepted:  # TODO exec -> open
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self._saveWorkingDirectory = filePath.parent

        return filePath, dialog.selectedNameFilter()

    def getExistingDirectoryPath(
        self, parent: QWidget, caption: str, initialDirectory: Path | None = None
    ) -> Path | None:
        dirPath = None

        dirName = QFileDialog.getExistingDirectory(
            parent,
            caption,
            str(initialDirectory or self.getOpenWorkingDirectory()),
            QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
        )

        if dirName:
            dirPath = Path(dirName)
            self.setOpenWorkingDirectory(dirPath)

        return dirPath
