from __future__ import annotations
from pathlib import Path

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QFileDialog, QTableView, QTreeView, QWidget

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ...model import MetadataPresenter
from ...model.data import (DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter,
                           DiffractionPatternPresenter)
from ...view import DataParametersView
from ..tree import SimpleTreeModel
from .dataset import DatasetController
from .dialogFactory import FileDialogFactory
from .file import DatasetFileController
from .metadata import MetadataController
from .patterns import PatternsController
from .tableModel import DataArrayTableModel


class DataParametersController(Observer):

    def __init__(self, settingsRegistry: SettingsRegistry,
                 datasetInputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                 datasetPresenter: DiffractionDatasetPresenter,
                 metadataPresenter: MetadataPresenter,
                 patternPresenter: DiffractionPatternPresenter, view: DataParametersView,
                 tableView: QTableView, fileDialogFactory: FileDialogFactory) -> None:
        self._settingsRegistry = settingsRegistry
        self._datasetInputOutputPresenter = datasetInputOutputPresenter
        self._datasetPresenter = datasetPresenter
        self._view = view
        self._tableView = tableView
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = SimpleTreeModel(datasetPresenter.getContentsTree())
        self._tableModel = DataArrayTableModel()
        self._fileController = DatasetFileController.createInstance(datasetInputOutputPresenter,
                                                                    view.filePage)
        self._metadataController = MetadataController.createInstance(metadataPresenter,
                                                                     view.metadataPage)
        self._patternsController = PatternsController.createInstance(datasetInputOutputPresenter,
                                                                     datasetPresenter,
                                                                     patternPresenter,
                                                                     view.patternsPage,
                                                                     fileDialogFactory)
        self._datasetController = DatasetController.createInstance(datasetInputOutputPresenter,
                                                                   datasetPresenter,
                                                                   view.datasetPage,
                                                                   fileDialogFactory)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       datasetInputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       metadataPresenter: MetadataPresenter,
                       patternPresenter: DiffractionPatternPresenter, view: DataParametersView,
                       tableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> DataParametersController:
        controller = cls(settingsRegistry, datasetInputOutputPresenter, datasetPresenter,
                         metadataPresenter, patternPresenter, view, tableView, fileDialogFactory)
        settingsRegistry.addObserver(controller)
        datasetPresenter.addObserver(controller)

        view.datasetPage.contentsView.treeView.setModel(controller._treeModel)
        view.datasetPage.contentsView.treeView.selectionModel().currentChanged.connect(
            controller._updateDataArrayInTableView)
        tableView.setModel(controller._tableModel)

        if datasetPresenter.isAssembled:
            controller._switchToDatasetView()

        return controller

    def _switchToDatasetView(self) -> None:
        self._view.setCurrentIndex(self._view.count() - 1)

    def _updateDataArrayInTableView(self, current: QModelIndex, previous: QModelIndex) -> None:
        names = list()
        nodeItem = current.internalPointer()

        while not nodeItem.isRoot:
            names.append(nodeItem.data(0))
            nodeItem = nodeItem.parentItem

        arrayPath = '/' + '/'.join(reversed(names))
        array = self._datasetPresenter.openArray(arrayPath)
        self._tableModel.setArray(array)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._datasetInputOutputPresenter.startProcessingDiffractionPatterns()
            self._switchToDatasetView()
        elif observable is self._datasetPresenter:
            self._treeModel.setRootNode(self._datasetPresenter.getContentsTree())
