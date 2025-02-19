from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QDir, QFileInfo, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView, QFileSystemModel, QWizardPage

from ptychodus.api.observer import Observable, Observer

from ....model.patterns import PatternsAPI
from ....view.patterns import OpenDatasetWizardFilesPage

from ...data import FileDialogFactory

logger = logging.getLogger(__name__)


class OpenDatasetWizardFilesViewController(Observer):
    def __init__(self, api: PatternsAPI, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._api = api
        self._page = OpenDatasetWizardFilesPage()
        self._file_dialog_factory = file_dialog_factory

        self._fileSystemModel = QFileSystemModel()
        self._fileSystemModel.setFilter(QDir.Filter.AllEntries | QDir.Filter.AllDirs)
        self._fileSystemModel.setNameFilterDisables(False)

        self._fileSystemProxyModel = QSortFilterProxyModel()
        self._fileSystemProxyModel.setSourceModel(self._fileSystemModel)

        self._page.directoryComboBox.addItem(str(file_dialog_factory.getOpenWorkingDirectory()))
        self._page.directoryComboBox.addItem(str(Path.home()))
        self._page.directoryComboBox.setEditable(True)
        self._page.directoryComboBox.textActivated.connect(self._handleDirectoryComboBoxActivated)

        self._page.fileSystemTableView.setModel(self._fileSystemProxyModel)
        self._page.fileSystemTableView.setSortingEnabled(True)
        self._page.fileSystemTableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._page.fileSystemTableView.verticalHeader().hide()
        self._page.fileSystemTableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._page.fileSystemTableView.doubleClicked.connect(
            self._handleFileSystemTableDoubleClicked
        )
        self._page.fileSystemTableView.selectionModel().currentChanged.connect(
            self._checkIfComplete
        )

        for fileFilter in api.getOpenFileFilterList():
            self._page.fileTypeComboBox.addItem(fileFilter)

        self._page.fileTypeComboBox.textActivated.connect(self._setNameFiltersInFileSystemModel)

        self._setRootPath(file_dialog_factory.getOpenWorkingDirectory())
        self._sync_model_to_view()
        api.addObserver(self)

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._fileSystemModel.setRootPath(str(rootPath))
        proxyIndex = self._fileSystemProxyModel.mapFromSource(index)
        self._page.fileSystemTableView.setRootIndex(proxyIndex)
        self._page.directoryComboBox.setCurrentText(str(rootPath))
        self._file_dialog_factory.setOpenWorkingDirectory(rootPath)

    def _handleDirectoryComboBoxActivated(self, text: str) -> None:
        fileInfo = QFileInfo(text)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def _handleFileSystemTableDoubleClicked(self, proxyIndex: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        fileInfo = self._fileSystemModel.fileInfo(index)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def openDataset(self) -> None:
        proxyIndex = self._page.fileSystemTableView.currentIndex()
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        filePath = Path(self._fileSystemModel.filePath(index))
        self._file_dialog_factory.setOpenWorkingDirectory(filePath.parent)

        fileFilter = self._page.fileTypeComboBox.currentText()
        self._api.openPatterns(filePath, fileType=fileFilter)

    def _checkIfComplete(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(current)
        fileInfo = self._fileSystemModel.fileInfo(index)
        self._page._setComplete(fileInfo.isFile())

    def _setNameFiltersInFileSystemModel(self, currentText: str) -> None:
        z = re.search(r'\((.+)\)', currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f'Dataset File Name Filters: {nameFilters}')
            self._fileSystemModel.setNameFilters(nameFilters)

    def _sync_model_to_view(self) -> None:
        self._page.fileTypeComboBox.setCurrentText(self._api.getOpenFileFilter())

    def update(self, observable: Observable) -> None:
        if observable is self._api:
            self._sync_model_to_view()

    def getWidget(self) -> QWizardPage:
        return self._page
