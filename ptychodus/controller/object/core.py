from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectRepositoryItem, ObjectRepositoryPresenter
from ...model.probe import ApparatusPresenter
from ...view.image import ImageView
from ...view.widgets import ExceptionDialog, RepositoryTreeView
from ..data import FileDialogFactory
from ..image import ImageController
from .compare import CompareObjectViewController
from .random import RandomObjectViewController
from .treeModel import ObjectTreeModel, ObjectTreeNode

logger = logging.getLogger(__name__)


class ObjectController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'  # TODO clean up

    def __init__(self, apparatusPresenter: ApparatusPresenter,
                 repositoryPresenter: ObjectRepositoryPresenter, imagePresenter: ImagePresenter,
                 view: RepositoryTreeView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._apparatusPresenter = apparatusPresenter
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = ObjectTreeModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, apparatusPresenter: ApparatusPresenter,
                       repositoryPresenter: ObjectRepositoryPresenter,
                       imagePresenter: ImagePresenter, view: RepositoryTreeView,
                       imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        controller = cls(apparatusPresenter, repositoryPresenter, imagePresenter, view, imageView,
                         fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)

        view.treeView.setModel(controller._treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.treeView.selectionModel().selectionChanged.connect(controller._updateView)

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.buttonBox.editButton.clicked.connect(controller._editSelectedObject)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedObject)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedObject)

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
            self._view,
            'Open Object',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openObject(filePath, nameFilter)

    def _saveSelectedObject(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Object',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.internalPointer().getName()

                try:
                    self._repositoryPresenter.saveObject(name, filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No items are selected!')

    def _editSelectedObject(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            itemPresenter = current.internalPointer().presenter  # TODO do this cleaner
            item = itemPresenter.item
            initializerName = item.getInitializerSimpleName()

            if initializerName == 'Random':
                RandomObjectViewController.editParameters(itemPresenter, self._view)
            elif initializerName == 'Compare':
                CompareObjectViewController.editParameters(itemPresenter, self._view)
            else:
                _ = QMessageBox.information(self._view, itemPresenter.name,
                                            f'\"{initializerName}\" has no editable parameters.')
        else:
            logger.error('No items are selected!')

    def _removeSelectedObject(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            name = current.internalPointer().getName()
            self._repositoryPresenter.removeObject(name)
        else:
            logger.error('No items are selected!')

    def _updateView(self) -> None:
        selectionModel = self._view.treeView.selectionModel()
        hasSelection = selectionModel.hasSelection()

        self._view.buttonBox.saveButton.setEnabled(hasSelection)
        self._view.buttonBox.editButton.setEnabled(hasSelection)
        self._view.buttonBox.removeButton.setEnabled(hasSelection)

        for index in selectionModel.selectedIndexes():
            node = index.internalPointer()
            pixelGeometry = self._apparatusPresenter.getObjectPlanePixelGeometry()
            self._imagePresenter.setArray(node.getArray(), pixelGeometry)

            return

        self._imagePresenter.clearArray()

    def _syncModelToView(self) -> None:
        for itemPresenter in self._repositoryPresenter:
            itemPresenter.item.addObserver(self)

        rootNode = ObjectTreeNode.createRoot()

        for itemPresenter in self._repositoryPresenter:
            rootNode.createChild(itemPresenter)

        self._treeModel.setRootNode(rootNode)

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
        elif isinstance(observable, ObjectRepositoryItem):
            for row, itemPresenter in enumerate(self._repositoryPresenter):
                if observable is itemPresenter.item:
                    self._treeModel.refreshObject(row)
                    self._updateView()
                    break
