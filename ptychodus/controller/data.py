from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

import numpy
from PyQt5.QtCore import Qt, QAbstractTableModel, QDir, QModelIndex, QObject, QVariant
from PyQt5.QtWidgets import QDialog, QFileDialog, QTableView, QTreeView, QWidget

from ..model import DataFilePresenter, Observable, Observer
from .tree import SimpleTreeModel


class FileDialogFactory:

    def __init__(self) -> None:
        self._openWorkingDirectory = Path.cwd()
        self._saveWorkingDirectory = Path.cwd()

    def getOpenFilePath(self,
                        parent: QWidget,
                        caption: str,
                        nameFilters: Optional[list[str]] = None,
                        mimeTypeFilters: Optional[list[str]] = None,
                        selectedNameFilter: Optional[str] = None) -> tuple[Optional[Path], str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self._openWorkingDirectory))
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        dialog.setFileMode(QFileDialog.ExistingFile)

        if nameFilters is not None:
            dialog.setNameFilters(nameFilters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selectedNameFilter is not None:
            dialog.selectNameFilter(selectedNameFilter)

        if dialog.exec_() == QDialog.Accepted:
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self._openWorkingDirectory = filePath.parent

        return filePath, dialog.selectedNameFilter()

    def getSaveFilePath(self,
                        parent: QWidget,
                        caption: str,
                        nameFilters: Optional[list[str]] = None,
                        mimeTypeFilters: Optional[list[str]] = None,
                        selectedNameFilter: Optional[str] = None) -> tuple[Optional[Path], str]:
        filePath = None

        dialog = QFileDialog(parent, caption, str(self._saveWorkingDirectory))
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.AnyFile)

        if nameFilters is not None:
            dialog.setNameFilters(nameFilters)

        if mimeTypeFilters is not None:
            dialog.setMimeTypeFilters(mimeTypeFilters)

        if selectedNameFilter is not None:
            dialog.selectNameFilter(selectedNameFilter)

        if dialog.exec_() == QDialog.Accepted:
            fileNameList = dialog.selectedFiles()
            fileName = fileNameList[0]

            if fileName:
                filePath = Path(fileName)
                self._saveWorkingDirectory = filePath.parent

        return filePath, dialog.selectedNameFilter()


class DataArrayTableModel(QAbstractTableModel):

    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self._array: Optional[numpy.ndarray] = None

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> Any:
        result = None

        if role == Qt.DisplayRole:
            result = section

        return QVariant(result)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        result = None

        if index.isValid() and role == Qt.DisplayRole and self._array is not None:
            result = str(self._array[index.row(), index.column()])

        return QVariant(result)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:
            count = self._array.shape[0]

        return count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:
            count = self._array.shape[1]

        return count

    def setArray(self, data: Optional[numpy.ndarray]) -> None:
        self.beginResetModel()
        self._array = None

        if data is not None:
            array = numpy.atleast_2d(data)
            self._array = array.T if numpy.ndim(data) == 1 else array

        self.endResetModel()


class DataFileController(Observer):

    def __init__(self, presenter: DataFilePresenter, treeView: QTreeView,
                 tableView: QTableView) -> None:
        self._presenter = presenter
        self._treeModel = SimpleTreeModel(presenter.getContentsTree())
        self._treeView = treeView
        self._tableModel = DataArrayTableModel()
        self._tableView = tableView

    @classmethod
    def createInstance(cls, presenter: DataFilePresenter, treeView: QTreeView,
                       tableView: QTableView) -> DataFileController:
        controller = cls(presenter, treeView, tableView)
        treeView.setModel(controller._treeModel)
        treeView.selectionModel().currentChanged.connect(controller.updateDataArrayInTableView)
        tableView.setModel(controller._tableModel)
        presenter.addObserver(controller)
        return controller

    def updateDataArrayInTableView(self, current: QModelIndex, previous: QModelIndex) -> None:
        names = list()
        nodeItem = current.internalPointer()

        while not nodeItem.isRoot:
            names.append(nodeItem.data(0))
            nodeItem = nodeItem.parentItem

        dataPath = '/' + '/'.join(reversed(names))
        data = self._presenter.openDataset(dataPath)
        self._tableModel.setArray(data)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._treeModel.setRootNode(self._presenter.getContentsTree())
