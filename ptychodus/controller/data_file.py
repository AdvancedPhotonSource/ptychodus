from pathlib import Path

import numpy
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from ..model import *
from .tree import SimpleTreeModel


class FileDialogFactory:
    def __init__(self) -> None:
        self._openWorkingDirectory = Path.home()
        self._saveWorkingDirectory = Path.home()

    def getOpenFilePath(self, parent: QWidget, caption: str, fileFilter: str) -> Optional[Path]:
        fileName, _ = QFileDialog.getOpenFileName(parent, caption, str(self._openWorkingDirectory),
                                                  fileFilter)
        # TODO update self._openWorkingDirectory
        return Path(fileName) if fileName else None

    def getSaveFilePath(self, parent: QWidget, caption: str, fileFilter: str) -> Optional[Path]:
        fileName, _ = QFileDialog.getSaveFileName(parent, caption, str(self._saveWorkingDirectory),
                                                  fileFilter)
        # TODO update self._saveWorkingDirectory
        return Path(fileName) if fileName else None


class DataArrayTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject = None) -> None:
        super().__init__(parent)
        self._array = None

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: Qt.ItemDataRole) -> QVariant:
        result = QVariant()

        if role == Qt.DisplayRole:
            result = section

        return result

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        result = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            result = str(self._array[index.row(), index.column()])

        return result

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:
            count = self._array.shape[0]

        return count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        count = 0

        if self._array is not None:  # and len(self._array.shape) > 1:
            count = self._array.shape[1]

        return count

    def setArray(self, data: numpy.ndarray) -> None:
        self.beginResetModel()
        self._array = None if data is None else numpy.atleast_2d(data)

        if numpy.ndim(data) == 1:
            self._array = self._array.T

        self.endResetModel()


class DataFileController:
    def __init__(self, presenter: DataFilePresenter, treeReader: H5FileTreeReader,
                 treeView: QTreeView, tableView: QTableView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._treeReader = treeReader
        self._treeModel = SimpleTreeModel(treeReader.getTree())
        self._treeView = treeView
        self._tableModel = DataArrayTableModel()
        self._tableView = tableView
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: DataFilePresenter, treeReader: H5FileTreeReader,
                       treeView: QTreeView, tableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> None:
        controller = cls(presenter, treeReader, treeView, tableView, fileDialogFactory)
        treeView.setModel(controller._treeModel)
        treeView.selectionModel().currentChanged.connect(controller.updateDataArrayInTableView)
        tableView.setModel(controller._tableModel)
        treeReader.addObserver(controller)
        return controller

    def openDataFile(self) -> None:
        filePath = self._fileDialogFactory.getOpenFilePath(self._treeView, 'Open Data File',
                                                           H5FileTreeReader.FILE_FILTER)

        if filePath:
            self._presenter.readFile(filePath)

    def updateDataArrayInTableView(self, current: QModelIndex, previous: QModelIndex) -> None:
        names = list()
        nodeItem = current.internalPointer()

        while not nodeItem.isRoot:
            names.append(nodeItem.data(0))
            nodeItem = nodeItem.parentItem

        dataPath = '/' + '/'.join(reversed(names))
        data = self._presenter.readData(dataPath)
        self._tableModel.setArray(data)

    def update(self, observable: Observable) -> None:
        if observable is self._treeReader:
            self._treeModel.setRootNode(self._treeReader.getTree())
