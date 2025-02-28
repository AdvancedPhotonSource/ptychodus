from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QDir, QFileInfo, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileSystemModel,
    QFormLayout,
    QTableView,
    QWizardPage,
)

from ptychodus.api.observer import Observable, Observer

from ....model.patterns import PatternsAPI
from ....view.patterns import OpenDatasetWizardPage

from ...data import FileDialogFactory

logger = logging.getLogger(__name__)


class OpenDatasetWizardFilesViewController(Observer):
    def __init__(self, api: PatternsAPI, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._api = api
        self._file_dialog_factory = file_dialog_factory

        self._location_combo_box = QComboBox()
        self._location_combo_box.addItem(str(file_dialog_factory.getOpenWorkingDirectory()))
        self._location_combo_box.addItem(str(Path.home()))
        self._location_combo_box.setEditable(True)
        self._location_combo_box.textActivated.connect(self._handleDirectoryComboBoxActivated)

        self._file_system_model = QFileSystemModel()
        self._file_system_model.setFilter(QDir.Filter.AllEntries | QDir.Filter.AllDirs)
        self._file_system_model.setNameFilterDisables(False)

        self._file_system_proxy_model = QSortFilterProxyModel()
        self._file_system_proxy_model.setSourceModel(self._file_system_model)

        self._file_system_table_view = QTableView()
        self._file_system_table_view.setModel(self._file_system_proxy_model)
        self._file_system_table_view.setSortingEnabled(True)
        self._file_system_table_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._file_system_table_view.verticalHeader().hide()
        self._file_system_table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._file_system_table_view.doubleClicked.connect(self._handleFileSystemTableDoubleClicked)
        self._file_system_table_view.selectionModel().currentChanged.connect(self._checkIfComplete)

        self._file_reader_chooser = api.getFileReaderChooser()
        self._file_reader_chooser.addObserver(self)

        self._file_type_combo_box = QComboBox()

        for fileFilter in self._file_reader_chooser.getDisplayNameList():
            self._file_type_combo_box.addItem(fileFilter)

        self._file_type_combo_box.textActivated.connect(self._setNameFiltersInFileSystemModel)

        layout = QFormLayout()
        # FIXME breadcrumbs
        layout.addRow('Location:', self._location_combo_box)
        layout.addRow(self._file_system_table_view)
        layout.addRow('File Type:', self._file_type_combo_box)

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Choose Dataset File(s)')
        self._page.setLayout(layout)

        # FIXME MOVE vvv?
        self._setRootPath(file_dialog_factory.getOpenWorkingDirectory())
        # FIXME MOVE ^^^?

        self._sync_model_to_view()

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._file_system_model.setRootPath(str(rootPath))
        proxyIndex = self._file_system_proxy_model.mapFromSource(index)
        self._file_system_table_view.setRootIndex(proxyIndex)
        self._location_combo_box.setCurrentText(str(rootPath))
        self._file_dialog_factory.setOpenWorkingDirectory(rootPath)

    def _handleDirectoryComboBoxActivated(self, text: str) -> None:
        file_info = QFileInfo(text)

        if file_info.isDir():
            self._setRootPath(Path(file_info.canonicalFilePath()))

    def _handleFileSystemTableDoubleClicked(self, proxyIndex: QModelIndex) -> None:
        index = self._file_system_proxy_model.mapToSource(proxyIndex)
        file_info = self._file_system_model.fileInfo(index)

        if file_info.isDir():
            self._setRootPath(Path(file_info.canonicalFilePath()))

    def openDataset(self) -> None:
        proxyIndex = self._file_system_table_view.currentIndex()
        index = self._file_system_proxy_model.mapToSource(proxyIndex)
        filePath = Path(self._file_system_model.filePath(index))
        self._file_dialog_factory.setOpenWorkingDirectory(filePath.parent)

        fileFilter = self._file_type_combo_box.currentText()
        self._api.openPatterns(filePath, fileType=fileFilter)

    def _checkIfComplete(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._file_system_proxy_model.mapToSource(current)
        file_info = self._file_system_model.fileInfo(index)
        self._page._setComplete(file_info.isFile())

    def _setNameFiltersInFileSystemModel(self, currentText: str) -> None:
        z = re.search(r'\((.+)\)', currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f'Dataset File Name Filters: {nameFilters}')
            self._file_system_model.setNameFilters(nameFilters)

    def _sync_model_to_view(self) -> None:
        self._file_type_combo_box.setCurrentText(
            self._file_reader_chooser.currentPlugin.displayName
        )

    def getWidget(self) -> QWizardPage:
        return self._page

    def update(self, observable: Observable) -> None:
        if observable is self._file_reader_chooser:
            self._sync_model_to_view()
