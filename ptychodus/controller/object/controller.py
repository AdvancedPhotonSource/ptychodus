from __future__ import annotations
from typing import Callable, Final, Optional
import logging

from PyQt5.QtCore import QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import (ObjectPresenter, ObjectRepositoryItemPresenter,
                             ObjectRepositoryPresenter, RandomObjectRepositoryItem)
from ...view import (ImageView, ObjectEditorDialog, ObjectParametersView, ObjectView,
                     RandomObjectView)
from ..data import FileDialogFactory
from ..image import ImageController
from .random import RandomObjectController
from .tableModel import ObjectTableModel

logger = logging.getLogger(__name__)


class ObjectParametersController(Observer):

    def __init__(self, presenter: ObjectRepositoryPresenter, view: ObjectParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ObjectRepositoryPresenter,
                       view: ObjectParametersView) -> ObjectParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.pixelSizeXWidget.setReadOnly(True)
        view.pixelSizeYWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.pixelSizeXWidget.setLengthInMeters(self._presenter.getPixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(self._presenter.getPixelSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ObjectController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'

    def __init__(self, repositoryPresenter: ObjectRepositoryPresenter,
                 imagePresenter: ImagePresenter, view: ObjectView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._parametersController = ObjectParametersController.createInstance(
            repositoryPresenter, view.parametersView)
        self._tableModel = ObjectTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, repositoryPresenter: ObjectRepositoryPresenter,
                       imagePresenter: ImagePresenter, view: ObjectView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        controller = cls(repositoryPresenter, imagePresenter, view, imageView, fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        view.repositoryView.tableView.setModel(controller._proxyModel)
        view.repositoryView.tableView.setSortingEnabled(True)
        view.repositoryView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.repositoryView.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        view.repositoryView.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._updateView())

        initializerNameList = repositoryPresenter.getInitializerNameList()
        initializerNameList.insert(0, ObjectController.OPEN_FILE)

        for name in initializerNameList:
            insertAction = view.repositoryView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.repositoryView.buttonBox.editButton.clicked.connect(controller._editSelectedObject)
        view.repositoryView.buttonBox.saveButton.clicked.connect(controller._saveSelectedObject)
        view.repositoryView.buttonBox.removeButton.clicked.connect(
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
            self._view.repositoryView,
            'Open Object',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openObject(filePath, nameFilter)

    def _saveSelectedObject(self) -> None:
        current = self._view.repositoryView.tableView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view.repositoryView,
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
        proxyIndex = self._view.repositoryView.tableView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]

        return itemPresenter

    def _editSelectedObject(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()

        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
            item = itemPresenter.item

            if isinstance(item, RandomObjectRepositoryItem):
                randomDialog = ObjectEditorDialog.createInstance(RandomObjectView.createInstance(),
                                                                 self._view)
                randomDialog.setWindowTitle(itemPresenter.name)
                randomController = RandomObjectController.createInstance(
                    item, randomDialog.editorView)
                randomDialog.open()
            else:
                logger.error('Unknown object repository item!')

    def _removeSelectedObject(self) -> None:
        current = self._view.repositoryView.tableView.currentIndex()

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
            array = itemPresenter.item.getArray()
            self._imagePresenter.setArray(array)

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._view.repositoryView.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._repositoryPresenter.canRemoveObject(name)

        self._view.repositoryView.buttonBox.saveButton.setEnabled(enable)
        self._view.repositoryView.buttonBox.editButton.setEnabled(enable)
        self._view.repositoryView.buttonBox.removeButton.setEnabled(enableRemove)

    def _updateView(self) -> None:
        self._setButtonsEnabled()
        self._setCurrentImage()

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()
        self._updateView()

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
