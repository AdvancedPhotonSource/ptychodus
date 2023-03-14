from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtCore import QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.object import ObjectPresenter, ObjectRepositoryPresenter
from ...view import ObjectParametersView
from ..data import FileDialogFactory
from .tableModel import ObjectTableModel

logger = logging.getLogger(__name__)


class ObjectController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'

    def __init__(self, presenter: ObjectPresenter, repositoryPresenter: ObjectRepositoryPresenter,
                 view: ObjectParametersView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._repositoryPresenter = repositoryPresenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ObjectTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter,
                       repositoryPresenter: ObjectRepositoryPresenter, view: ObjectParametersView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        controller = cls(presenter, repositoryPresenter, view, fileDialogFactory)
        presenter.addObserver(controller)
        repositoryPresenter.addObserver(controller)

        view.objectView.pixelSizeXWidget.setReadOnly(True)
        view.objectView.pixelSizeYWidget.setReadOnly(True)

        controller._proxyModel.setSourceModel(controller._tableModel)
        view.estimatesView.tableView.setModel(controller._proxyModel)
        view.estimatesView.tableView.setSortingEnabled(True)
        view.estimatesView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.estimatesView.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        view.estimatesView.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._setButtonsEnabled())

        itemNameList = presenter.getItemNameList()
        itemNameList.insert(0, ObjectController.OPEN_FILE)

        for name in itemNameList:
            insertAction = view.estimatesView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.estimatesView.buttonBox.editButton.clicked.connect(controller._editSelectedObject)
        view.estimatesView.buttonBox.saveButton.clicked.connect(controller._saveSelectedObject)
        view.estimatesView.buttonBox.removeButton.clicked.connect(controller._removeSelectedObject)

        # FIXME view selected object
        controller._syncModelToView()

        return controller

    def _initializeObject(self, name: str) -> None:
        if name == ObjectController.OPEN_FILE:
            self._openObject()
        else:
            self._presenter.initializeObject(name)

    def _createItemLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeObject(name)

    def _openObject(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view.estimatesView,
            'Open Object',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openObject(filePath, nameFilter)

    def _saveSelectedObject(self) -> None:
        current = self._view.estimatesView.tableView.selectionModel().currentIndex()

        if not current.isValid():
            logger.error('No objects are selected!')
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view.estimatesView,
            'Save Object',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            name = current.sibling(current.row(), 0).data()
            self._presenter.saveObject(name, filePath, nameFilter)

    def _editSelectedObject(self) -> None:
        current = self._view.estimatesView.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            #category = current.sibling(current.row(), 1).data()
            #item = self._presenter.getItem(name)

            # FIXME edit object
            #if isinstance(item._item, CartesianObjectRepositoryItem):
            #    cartesianDialog = ObjectEditorDialog.createInstance(
            #        CartesianObjectView.createInstance(), self._parametersView)
            #    cartesianDialog.setWindowTitle(name)
            #    cartesianController = CartesianObjectController.createInstance(
            #        item._item, cartesianDialog.editorView)
            #    cartesianTransformController = ObjectTransformController.createInstance(
            #        item, cartesianDialog.transformView)
            #    cartesianDialog.open()
            #else:
            #    logger.debug(f'Unknown category \"{category}\"')
        else:
            logger.error('No objects are selected!')

    def _removeSelectedObject(self) -> None:
        current = self._view.estimatesView.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeObject(name)
        else:
            logger.error('No objects are selected!')

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._view.estimatesView.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._repositoryPresenter.canRemoveObject(name)

        self._view.estimatesView.buttonBox.saveButton.setEnabled(enable)
        self._view.estimatesView.buttonBox.editButton.setEnabled(enable)
        self._view.estimatesView.buttonBox.removeButton.setEnabled(enableRemove)

    def _syncModelToView(self) -> None:
        self._view.objectView.pixelSizeXWidget.setLengthInMeters(
            self._presenter.getPixelSizeXInMeters())
        self._view.objectView.pixelSizeYWidget.setLengthInMeters(
            self._presenter.getPixelSizeYInMeters())

        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

        self._setButtonsEnabled()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
        elif observable is self._repositoryPresenter:
            self._syncModelToView()
