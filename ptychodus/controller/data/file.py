from __future__ import annotations
from pathlib import Path
import logging
import re

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QFileSystemModel

from ...model import DiffractionDatasetPresenter, Observable, Observer
from ...view import DataNavigationPage, DatasetFileView
from .dialogFactory import FileDialogFactory

logger = logging.getLogger(__name__)


class DatasetFileController(Observer):

    def __init__(self, presenter: DiffractionDatasetPresenter,
                 view: DataNavigationPage[DatasetFileView],
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._fileSystemModel = QFileSystemModel()

    @classmethod
    def createInstance(cls, presenter: DiffractionDatasetPresenter,
                       view: DataNavigationPage[DatasetFileView],
                       fileDialogFactory: FileDialogFactory) -> DatasetFileController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        for fileFilter in presenter.getOpenFileFilterList():
            view.contentsView.fileTypeComboBox.addItem(fileFilter)

        rootPath = controller._fileSystemModel.setRootPath(
            str(fileDialogFactory.getOpenWorkingDirectory()))
        controller._fileSystemModel.setNameFilterDisables(False)
        view.contentsView.fileSystemTreeView.setModel(controller._fileSystemModel)
        view.contentsView.fileSystemTreeView.selectionModel().currentChanged.connect(
            controller._updateEnabledNavigationButtons)

        view.forwardButton.clicked.connect(controller._openDiffractionFile)

        controller._syncModelToView()

        view.contentsView.fileTypeComboBox.currentTextChanged.connect(
            controller._setNameFiltersInFileSystemModel)
        controller._setNameFiltersInFileSystemModel(
            view.contentsView.fileTypeComboBox.currentText())

        view.backwardButton.setVisible(False)
        view.forwardButton.setEnabled(False)

        return controller

    def _convertModelIndexToPath(self, index: QModelIndex) -> Path:
        pathParts = list()
        node = index

        while node.isValid():
            pathParts.append(node.data())
            node = node.parent()

        pathParts.reverse()

        return Path(pathParts[0] + '/'.join(pathParts[1:]))

    def _openDiffractionFile(self) -> None:
        filePath = self._convertModelIndexToPath(
            self._view.contentsView.fileSystemTreeView.currentIndex())
        fileFilter = self._view.contentsView.fileTypeComboBox.currentText()
        print(f'{filePath} :: {fileFilter}')
        self._presenter.openDiffractionFile(filePath, fileFilter)
        # FIXME self._view.datasetPage.contentsView.setTitle(f'Data File: {filePath}')

    def _updateEnabledNavigationButtons(self, current: QModelIndex, previous: QModelIndex) -> None:
        currentPath = self._convertModelIndexToPath(current)
        self._view.forwardButton.setEnabled(currentPath.is_file())

    def _syncModelToView(self) -> None:
        openFileFilter = self._presenter.getOpenFileFilter()
        self._view.contentsView.fileTypeComboBox.setCurrentText(openFileFilter)

    def _setNameFiltersInFileSystemModel(self, currentText) -> None:
        z = re.search('\((.+)\)', currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f'Dataset File Name Filters: {nameFilters}')
            self._fileSystemModel.setNameFilters(nameFilters)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
