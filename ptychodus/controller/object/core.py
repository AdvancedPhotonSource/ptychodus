from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel, Qt
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QMessageBox

from ...api.observer import SequenceObserver
from ...model.image import ImagePresenter
from ...model.object import ObjectRepositoryItem
from ...model.product import ObjectRepository
from ...view.image import ImageView
from ...view.repository import RepositoryItemCopierDialog, RepositoryTreeView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
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

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in repository.builderNames()])
        builderItemDelegate = ComboBoxItemDelegate(builderListModel, view.treeView)

        view.treeView.setModel(treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.setItemDelegateForColumn(2, builderItemDelegate)
        view.treeView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        loadFromFileAction = view.buttonBox.loadMenu.addAction('Open File...')
        loadFromFileAction.triggered.connect(controller._loadCurrentObjectFromFile)

        copyAction = view.buttonBox.loadMenu.addAction('Copy...')
        copyAction.triggered.connect(controller._copyToCurrentObject)

        view.copierDialog.setWindowTitle('Object Copier')
        view.copierDialog.sourceComboBox.setModel(treeModel)
        view.copierDialog.destinationComboBox.setModel(treeModel)
        view.copierDialog.finished.connect(controller._finishCopyingObject)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentObject)
        view.buttonBox.saveButton.clicked.connect(controller._saveCurrentObject)

        compareAction = view.buttonBox.analyzeMenu.addAction('Compare...')
        compareAction.triggered.connect(controller._compareCurrentObject)

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

    def _loadCurrentObjectFromFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Object',
            nameFilters=self._repository.getOpenFileFilterList(),
            selectedNameFilter=self._repository.getOpenFileFilter())

        if filePath:
            try:
                self._repository.openObject(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Reader', err)

    def _copyToCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex >= 0:
            self._view.copierDialog.destinationComboBox.setCurrentIndex(itemIndex)
            self._view.copierDialog.open()

    def _finishCopyingObject(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            sourceIndex = self._view.copierDialog.sourceComboBox.currentIndex()
            destinationIndex = self._view.copierDialog.destinationComboBox.currentIndex()
            self._repository.copyObject(sourceIndex, destinationIndex)

    def _editCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        print(f'Edit {itemIndex}')  # FIXME

    def _saveCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Object',
            nameFilters=self._repository.getSaveFileFilterList(),
            selectedNameFilter=self._repository.getSaveFileFilter())

        if filePath:
            try:
                self._repository.saveObject(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _compareCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        print(f'Compare {itemIndex}')  # FIXME

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.loadButton.setEnabled(enabled)
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imagePresenter.clearArray()
        else:
            try:
                item = self._repository[itemIndex]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                object_ = item.getObject()
                self._imagePresenter.setArray(object_.array, object_.getPixelGeometry())

    def handleItemInserted(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.insertItem(index, item)

    def handleItemChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.updateItem(index, item)

    def handleItemRemoved(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.removeItem(index, item)
