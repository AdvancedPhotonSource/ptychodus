from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtCore import QItemSelection, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectRepositoryItemPresenter, ObjectRepositoryPresenter
from ...model.probe import ApparatusPresenter
from ...view.image import ImageView
from ...view.object import ObjectParametersView, ObjectView
from ..data import FileDialogFactory
from ..image import ImageController
from .random import RandomObjectViewController
from .tableModel import ObjectTableModel

logger = logging.getLogger(__name__)


class ObjectParametersController(Observer):

    def __init__(self, presenter: ApparatusPresenter, view: ObjectParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ApparatusPresenter,
                       view: ObjectParametersView) -> ObjectParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.pixelSizeXWidget.setReadOnly(True)
        view.pixelSizeYWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.pixelSizeXWidget.setLengthInMeters(
            self._presenter.getObjectPlanePixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(
            self._presenter.getObjectPlanePixelSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ObjectController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'  # TODO clean up

    def __init__(self, apparatusPresenter: ApparatusPresenter,
                 repositoryPresenter: ObjectRepositoryPresenter, imagePresenter: ImagePresenter,
                 view: ObjectView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._parametersController = ObjectParametersController.createInstance(
            apparatusPresenter, view.parametersView)
        self._tableModel = ObjectTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, apparatusPresenter: ApparatusPresenter,
                       repositoryPresenter: ObjectRepositoryPresenter,
                       imagePresenter: ImagePresenter, view: ObjectView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        controller = cls(apparatusPresenter, repositoryPresenter, imagePresenter, view, imageView,
                         fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        # FIXME remove active probe then cannot make valid again
        # FIXME save probe without suffix then get exception because something adds the suffix during save
        # FIXME save/load from restart file
        # FIXME need to switch current scan/probe/object for reconstruction

        controller._proxyModel.setSourceModel(controller._tableModel)
        view.repositoryView.tableView.setModel(controller._proxyModel)
        view.repositoryView.tableView.setSortingEnabled(True)
        view.repositoryView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.repositoryView.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        view.repositoryView.tableView.selectionModel().selectionChanged.connect(
            controller._updateView)

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.repositoryView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.repositoryView.buttonBox.editButton.clicked.connect(controller._editSelectedObject)
        view.repositoryView.buttonBox.saveButton.clicked.connect(controller._saveSelectedObject)
        view.repositoryView.buttonBox.removeButton.clicked.connect(
            controller._removeSelectedObject)

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

    def _getSelectedItemPresenter(self) -> ObjectRepositoryItemPresenter | None:
        itemPresenter: ObjectRepositoryItemPresenter | None = None
        proxyIndex = self._view.repositoryView.tableView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]

        return itemPresenter

    def _editSelectedObject(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()  # FIXME do this differently

        # FIXME update while editing
        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
            item = itemPresenter.item
            initializerName = item.getInitializerSimpleName()

            if initializerName == 'Random':
                randomController = RandomObjectViewController.createInstance(
                    itemPresenter, self._view)
                randomController.openDialog()
            else:
                # FIXME FromFile
                logger.error('Unknown object repository item!')

    def _removeSelectedObject(self) -> None:
        current = self._view.repositoryView.tableView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeObject(name)
        else:
            logger.error('No items are selected!')

    def _updateView(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        for index in deselected.indexes():
            self._view.repositoryView.buttonBox.saveButton.setEnabled(False)
            self._view.repositoryView.buttonBox.editButton.setEnabled(False)
            self._view.repositoryView.buttonBox.removeButton.setEnabled(False)

            self._imagePresenter.clearArray()
            break

        for index in selected.indexes():
            self._view.repositoryView.buttonBox.saveButton.setEnabled(True)
            self._view.repositoryView.buttonBox.editButton.setEnabled(True)
            self._view.repositoryView.buttonBox.removeButton.setEnabled(True)

            sourceIndex = self._proxyModel.mapToSource(index)
            itemPresenter = self._repositoryPresenter[sourceIndex.row()]
            array = itemPresenter.item.getArray()
            self._imagePresenter.setArray(array)
            break

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
