from __future__ import annotations
from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QDir, QFileInfo, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView, QFileSystemModel

from ...api.observer import Observable, Observer
from ...model.data import DiffractionDatasetInputOutputPresenter
from ...view.data import DataNavigationPage, DatasetFileView
from .dialogFactory import FileDialogFactory

logger = logging.getLogger(__name__)


class DatasetFileController(Observer):

    def __init__(self, presenter: DiffractionDatasetInputOutputPresenter,
                 view: DataNavigationPage[DatasetFileView],
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._fileSystemModel = QFileSystemModel()
        self._fileSystemProxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, presenter: DiffractionDatasetInputOutputPresenter,
                       view: DataNavigationPage[DatasetFileView],
                       fileDialogFactory: FileDialogFactory) -> DatasetFileController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        controller._fileSystemModel.setFilter(QDir.AllEntries | QDir.AllDirs)
        controller._fileSystemModel.setNameFilterDisables(False)
        controller._fileSystemProxyModel.setSourceModel(controller._fileSystemModel)
        view.contentsView.fileSystemTableView.setModel(controller._fileSystemProxyModel)
        view.contentsView.fileSystemTableView.setSortingEnabled(True)
        view.contentsView.fileSystemTableView.sortByColumn(0, Qt.AscendingOrder)
        view.contentsView.fileSystemTableView.verticalHeader().hide()
        view.contentsView.fileSystemTableView.setSelectionBehavior(QAbstractItemView.SelectRows)

        for fileFilter in presenter.getOpenFileFilterList():
            view.contentsView.fileTypeComboBox.addItem(fileFilter)

        view.contentsView.fileTypeComboBox.currentTextChanged.connect(
            controller._setNameFiltersInFileSystemModel)
        controller._syncModelToView()

        view.contentsView.directoryComboBox.addItem(
            str(fileDialogFactory.getOpenWorkingDirectory()))
        view.contentsView.directoryComboBox.addItem(str(Path.home()))
        view.contentsView.directoryComboBox.setEditable(True)
        view.contentsView.directoryComboBox.textActivated.connect(
            controller._handleDirectoryComboBoxActivated)
        controller._setRootPath(fileDialogFactory.getOpenWorkingDirectory())

        view.contentsView.fileSystemTableView.doubleClicked.connect(
            controller._handleFileSystemTableDoubleClicked)
        view.contentsView.fileSystemTableView.selectionModel().currentChanged.connect(
            controller._updateEnabledNavigationButtons)

        view.backwardButton.setVisible(False)
        view.forwardButton.clicked.connect(controller._openDiffractionFile)
        view.forwardButton.setEnabled(False)

        return controller

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._fileSystemModel.setRootPath(str(rootPath))
        proxyIndex = self._fileSystemProxyModel.mapFromSource(index)
        self._view.contentsView.fileSystemTableView.setRootIndex(proxyIndex)
        self._view.contentsView.directoryComboBox.setCurrentText(str(rootPath))
        self._fileDialogFactory.setOpenWorkingDirectory(rootPath)

    def _handleDirectoryComboBoxActivated(self, text: str) -> None:
        fileInfo = QFileInfo(text)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def _handleFileSystemTableDoubleClicked(self, proxyIndex: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        fileInfo = self._fileSystemModel.fileInfo(index)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def _openDiffractionFile(self) -> None:
        proxyIndex = self._view.contentsView.fileSystemTableView.currentIndex()
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        filePath = Path(self._fileSystemModel.filePath(index))
        self._fileDialogFactory.setOpenWorkingDirectory(filePath.parent)

        fileFilter = self._view.contentsView.fileTypeComboBox.currentText()
        self._presenter.openDiffractionFile(filePath, fileFilter)

    def _updateEnabledNavigationButtons(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(current)
        fileInfo = self._fileSystemModel.fileInfo(index)
        self._view.forwardButton.setEnabled(fileInfo.isFile())

    def _setNameFiltersInFileSystemModel(self, currentText: str) -> None:
        z = re.search('\((.+)\)', currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f'Dataset File Name Filters: {nameFilters}')
            self._fileSystemModel.setNameFilters(nameFilters)

    def _syncModelToView(self) -> None:
        self._view.contentsView.fileTypeComboBox.setCurrentText(
            self._presenter.getOpenFileFilter())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
