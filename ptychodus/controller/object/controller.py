from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtCore import QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectPresenter, ObjectRepositoryPresenter
from ...view import ObjectParametersView, ImageView
from ..data import FileDialogFactory
from ..image import ImageController
from .tableModel import ObjectTableModel

logger = logging.getLogger(__name__)


class ObjectController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'

    def __init__(self, repositoryPresenter: ObjectRepositoryPresenter,
                 imagePresenter: ImagePresenter, parametersView: ObjectParametersView,
                 imageView: ImageView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._parametersView = parametersView
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ObjectTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, repositoryPresenter: ObjectRepositoryPresenter,
                       imagePresenter: ImagePresenter, parametersView: ObjectParametersView,
                       imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        controller = cls(repositoryPresenter, imagePresenter, parametersView, imageView,
                         fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        parametersView.objectView.pixelSizeXWidget.setReadOnly(True)
        parametersView.objectView.pixelSizeYWidget.setReadOnly(True)

        controller._proxyModel.setSourceModel(controller._tableModel)
        parametersView.estimatesView.tableView.setModel(controller._proxyModel)
        parametersView.estimatesView.tableView.setSortingEnabled(True)
        parametersView.estimatesView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        parametersView.estimatesView.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        parametersView.estimatesView.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._updateView())

        itemNameList = repositoryPresenter.getItemNameList()
        itemNameList.insert(0, ObjectController.OPEN_FILE)

        for name in itemNameList:
            insertAction = parametersView.estimatesView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        parametersView.estimatesView.buttonBox.editButton.clicked.connect(
            controller._editSelectedObject)
        parametersView.estimatesView.buttonBox.saveButton.clicked.connect(
            controller._saveSelectedObject)
        parametersView.estimatesView.buttonBox.removeButton.clicked.connect(
            controller._removeSelectedObject)
        imageView.imageRibbon.indexGroupBox.setVisible(False)

        controller._syncModelToView()

        return controller

    def _initializeObject(self, name: str) -> None:
        if name == ObjectController.OPEN_FILE:
            self._openObject()
        else:
            self._repositoryPresenter.initializeObject(name)

    def _createItemLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeObject(name)

    def _openObject(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._parametersView.estimatesView,
            'Open Object',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openObject(filePath, nameFilter)

    def _saveSelectedObject(self) -> None:
        current = self._parametersView.estimatesView.tableView.currentIndex()

        if not current.isValid():
            logger.error('No objects are selected!')
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._parametersView.estimatesView,
            'Save Object',
            nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

        if filePath:
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.saveObject(name, filePath, nameFilter)

    def _editSelectedObject(self) -> None:
        current = self._parametersView.estimatesView.tableView.currentIndex()

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
        current = self._parametersView.estimatesView.tableView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeObject(name)
        else:
            logger.error('No objects are selected!')

    def _setCurrentImage(self) -> None:
        current = self._parametersView.estimatesView.tableView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            array = self._repositoryPresenter.getObjectArray(name)  # FIXME
            self._imagePresenter.setArray(array)
        else:
            logger.error('No objects are selected!')

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._parametersView.estimatesView.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._repositoryPresenter.canRemoveObject(name)

        self._parametersView.estimatesView.buttonBox.saveButton.setEnabled(enable)
        self._parametersView.estimatesView.buttonBox.editButton.setEnabled(enable)
        self._parametersView.estimatesView.buttonBox.removeButton.setEnabled(enableRemove)

    def _updateView(self) -> None:
        self._setButtonsEnabled()
        self._setCurrentImage()

    def _syncModelToView(self) -> None:
        self._parametersView.objectView.pixelSizeXWidget.setLengthInMeters(
            self._repositoryPresenter.getPixelSizeXInMeters())
        self._parametersView.objectView.pixelSizeYWidget.setLengthInMeters(
            self._repositoryPresenter.getPixelSizeYInMeters())

        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

        self._updateView()

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
