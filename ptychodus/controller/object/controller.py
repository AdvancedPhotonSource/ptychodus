from __future__ import annotations
from typing import Callable, Final, Optional
import logging

from PyQt5.QtCore import QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import (ObjectPresenter, ObjectRepositoryItemPresenter,
                             ObjectRepositoryPresenter)
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
        parametersView.repositoryWidget.tableView.setModel(controller._proxyModel)
        parametersView.repositoryWidget.tableView.setSortingEnabled(True)
        parametersView.repositoryWidget.tableView.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        parametersView.repositoryWidget.tableView.setSelectionMode(
            QAbstractItemView.SingleSelection)
        parametersView.repositoryWidget.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._updateView())

        initializerNameList = repositoryPresenter.getInitializerNameList()
        initializerNameList.insert(0, ObjectController.OPEN_FILE)

        for name in initializerNameList:
            insertAction = parametersView.repositoryWidget.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        parametersView.repositoryWidget.buttonBox.editButton.clicked.connect(
            controller._editSelectedObject)
        parametersView.repositoryWidget.buttonBox.saveButton.clicked.connect(
            controller._saveSelectedObject)
        parametersView.repositoryWidget.buttonBox.removeButton.clicked.connect(
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
            self._parametersView.repositoryWidget,
            'Open Object',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openObject(filePath, nameFilter)

    def _saveSelectedObject(self) -> None:
        current = self._parametersView.repositoryWidget.tableView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._parametersView.repositoryWidget,
                'Save Object',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.sibling(current.row(), 0).data()
                self._repositoryPresenter.saveObject(name, filePath, nameFilter)
        else:
            logger.error('No items are selected!')

    def _getSelectedItemPresenter(self) -> Optional[ObjectRepositoryItemPresenter]:
        itemPresenter: Optional[ObjectRepositoryItemPresenter] = None
        proxyIndex = self._parametersView.repositoryWidget.tableView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]

        return itemPresenter

    def _editSelectedObject(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()

        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
            #item = itemPresenter._item

            #if isinstance(item, CartesianObjectRepositoryItem):
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
            pass  # FIXME edit object

    def _removeSelectedObject(self) -> None:
        current = self._parametersView.repositoryWidget.tableView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeObject(name)
        else:
            logger.error('No items are selected!')

    def _setCurrentImage(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()

        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
            array = itemPresenter.getArray()
            self._imagePresenter.setArray(array)

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._parametersView.repositoryWidget.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._repositoryPresenter.canRemoveObject(name)

        self._parametersView.repositoryWidget.buttonBox.saveButton.setEnabled(enable)
        self._parametersView.repositoryWidget.buttonBox.editButton.setEnabled(enable)
        self._parametersView.repositoryWidget.buttonBox.removeButton.setEnabled(enableRemove)

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
