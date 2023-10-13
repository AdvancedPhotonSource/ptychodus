from __future__ import annotations

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QTableView

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ...model import MetadataPresenter
from ...model.data import (DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter,
                           DiffractionPatternPresenter)
from ...view.data import DataParametersView
from ..tree import SimpleTreeModel
from .dialogFactory import FileDialogFactory
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

        view.treeView.setModel(controller._treeModel)
        view.treeView.selectionModel().currentChanged.connect(
            controller._updateDataArrayInTableView)
        tableView.setModel(controller._tableModel)

        return controller

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
        if observable is self._settingsRegistry:  # FIXME relocate?
            self._datasetInputOutputPresenter.startAssemblingDiffractionPatterns()
        elif observable is self._datasetPresenter:
            self._treeModel.setRootNode(self._datasetPresenter.getContentsTree())
