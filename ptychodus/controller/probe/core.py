from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.probe import ApparatusPresenter, ProbeRepositoryItem, ProbeRepositoryPresenter
from ...view.image import ImageView
from ...view.repository import RepositoryTreeView
from ...view.widgets import ExceptionDialog, ProgressBarItemDelegate
from ..data import FileDialogFactory
from ..image import ImageController
from .disk import DiskProbeViewController
from .fzp import FresnelZonePlateProbeViewController
from .superGaussian import SuperGaussianProbeViewController
from .treeModel import ProbeTreeModel, ProbeTreeNode

logger = logging.getLogger(__name__)


class ProbeController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'  # TODO clean up

    def __init__(self, apparatusPresenter: ApparatusPresenter,
                 repositoryPresenter: ProbeRepositoryPresenter, imagePresenter: ImagePresenter,
                 view: RepositoryTreeView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._apparatusPresenter = apparatusPresenter
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = ProbeTreeModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, apparatusPresenter: ApparatusPresenter,
                       repositoryPresenter: ProbeRepositoryPresenter,
                       imagePresenter: ImagePresenter, view: RepositoryTreeView,
                       imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeController:
        controller = cls(apparatusPresenter, repositoryPresenter, imagePresenter, view, imageView,
                         fileDialogFactory)
        repositoryPresenter.addObserver(controller)
        delegate = ProgressBarItemDelegate(view.treeView)
        view.treeView.setItemDelegateForColumn(1, delegate)
        view.treeView.setModel(controller._treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.selectionModel().selectionChanged.connect(controller._updateView)

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.buttonBox.editButton.clicked.connect(controller._editSelectedProbe)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedProbe)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedProbe)

        controller._syncModelToView()

        return controller

    def _initializeProbe(self, name: str) -> None:
        if name == ProbeController.OPEN_FILE:
            self._openProbe()
        else:
            self._repositoryPresenter.initializeProbe(name)

    def _createItemLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeProbe(name)

    def _openProbe(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Probe',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openProbe(filePath, nameFilter)

    def _saveSelectedProbe(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Probe',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.internalPointer().getName()

                try:
                    self._repositoryPresenter.saveProbe(name, filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No items are selected!')

    def _editSelectedProbe(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            itemPresenter = current.internalPointer().presenter  # TODO do this cleaner
            item = itemPresenter.item
            initializerName = item.getInitializerSimpleName()

            if initializerName == 'Disk':
                DiskProbeViewController.editParameters(itemPresenter, self._view)
            elif initializerName == 'FresnelZonePlate':
                FresnelZonePlateProbeViewController.editParameters(itemPresenter, self._view)
            elif initializerName == 'SuperGaussian':
                SuperGaussianProbeViewController.editParameters(itemPresenter, self._view)
            else:
                _ = QMessageBox.information(self._view, itemPresenter.name,
                                            f'\"{initializerName}\" has no editable parameters.')
        else:
            logger.error('No items are selected!')

    def _removeSelectedProbe(self) -> None:
        current = self._view.treeView.currentIndex()

        if current.isValid():
            name = current.internalPointer().getName()
            self._repositoryPresenter.removeProbe(name)
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

        rootNode = ProbeTreeNode.createRoot()

        for itemPresenter in self._repositoryPresenter:
            rootNode.createChild(itemPresenter)

        self._treeModel.setRootNode(rootNode)

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
        elif isinstance(observable, ProbeRepositoryItem):
            for row, itemPresenter in enumerate(self._repositoryPresenter):
                if observable is itemPresenter.item:
                    self._treeModel.refreshProbe(row)
                    self._updateView()
                    break
