from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from ...api.observer import SequenceObserver
from ...model.image import ImagePresenter
from ...model.object import ObjectRepositoryItem
from ...model.product import ObjectRepository
from ...view.image import ImageView
from ...view.repository import RepositoryTreeView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .random import RandomObjectViewController
from .treeModel import ObjectTreeModel

logger = logging.getLogger(__name__)


class ObjectController(SequenceObserver[ObjectRepositoryItem]):

    def __init__(self, repository: ObjectRepository, imagePresenter: ImagePresenter,
                 view: RepositoryTreeView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel) -> None:
        super().__init__()
        self._repository = repository
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = treeModel
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, repository: ObjectRepository, imagePresenter: ImagePresenter,
                       view: RepositoryTreeView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        treeModel = ObjectTreeModel(repository)

        controller = cls(repository, imagePresenter, view, imageView, fileDialogFactory, treeModel)
        repository.addObserver(controller)

        view.treeView.setModel(treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        # FIXME populate build menu

        view.buttonBox.editButton.clicked.connect(controller._editSelectedObject)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedObject)
        view.buttonBox.analyzeButton.clicked.connect(controller._analyzeSelectedObject)

        return controller

    def _getCurrentItemIndex(self) -> int:
        modelIndex = self._view.treeView.currentIndex()

        if modelIndex.isValid():
            parent = modelIndex.parent()

            while parent.isValid():
                modelIndex = parent
                parent = modelIndex.parent()

            return modelIndex.row()

        logger.warning('No items are selected!')
        return -1

    def _buildSelectedObject(self) -> None:  # FIXME to treeView comboBox
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Object',
            nameFilters=self._repository.getOpenFileFilterList(),
            selectedNameFilter=self._repository.getOpenFileFilter())

        if not filePath:
            return

        try:
            self._repository.openObject(itemIndex, filePath, nameFilter)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('File Writer', err)

    def _editSelectedObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        print(f'Edit {itemIndex}')  # FIXME

    def _saveSelectedObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Object',
            nameFilters=self._repository.getSaveFileFilterList(),
            selectedNameFilter=self._repository.getSaveFileFilter())

        if not filePath:
            return

        try:
            self._repository.saveObject(itemIndex, filePath, nameFilter)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('File Writer', err)

    def _analyzeSelectedObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        print(f'Analyze {itemIndex}')  # FIXME

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imagePresenter.clearArray()
        else:
            pass  # FIXME self._imagePresenter.setArray(array, pixelGeometry)

    def handleItemInserted(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.insertItem(index, item)

    def handleItemChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.updateItem(index, item)

    def handleItemRemoved(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.removeItem(index, item)
