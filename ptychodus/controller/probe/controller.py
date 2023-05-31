from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.probe import (ApparatusPresenter, ProbeRepositoryItem, ProbeRepositoryPresenter)
from ...view.image import ImageView
from ...view.probe import ProbeParametersView, ProbeView
from ...view.widgets import ProgressBarItemDelegate
from ..data import FileDialogFactory
from ..image import ImageController
from .disk import DiskProbeViewController
from .fzp import FresnelZonePlateProbeViewController
from .superGaussian import SuperGaussianProbeViewController
from .treeModel import ProbeTreeModel, ProbeTreeNode

logger = logging.getLogger(__name__)


class ProbeParametersController(Observer):

    def __init__(self, presenter: ApparatusPresenter, view: ProbeParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ApparatusPresenter,
                       view: ProbeParametersView) -> ProbeParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)

        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.energyWidget.setEnergyInElectronVolts(
            self._presenter.getProbeEnergyInElectronVolts())
        self._view.wavelengthWidget.setLengthInMeters(self._presenter.getProbeWavelengthInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'  # TODO clean up

    def __init__(self, apparatusPresenter: ApparatusPresenter,
                 repositoryPresenter: ProbeRepositoryPresenter, imagePresenter: ImagePresenter,
                 view: ProbeView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._parametersController = ProbeParametersController.createInstance(
            apparatusPresenter, view.parametersView)
        self._treeModel = ProbeTreeModel()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, apparatusPresenter: ApparatusPresenter,
                       repositoryPresenter: ProbeRepositoryPresenter,
                       imagePresenter: ImagePresenter, view: ProbeView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeController:
        controller = cls(apparatusPresenter, repositoryPresenter, imagePresenter, view, imageView,
                         fileDialogFactory)
        repositoryPresenter.addObserver(controller)
        delegate = ProgressBarItemDelegate(view.repositoryView.treeView)
        view.repositoryView.treeView.setItemDelegateForColumn(1, delegate)
        view.repositoryView.treeView.setModel(controller._treeModel)
        view.repositoryView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.repositoryView.treeView.selectionModel().selectionChanged.connect(
            controller._updateView)

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.repositoryView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.repositoryView.buttonBox.editButton.clicked.connect(controller._editSelectedProbe)
        view.repositoryView.buttonBox.saveButton.clicked.connect(controller._saveSelectedProbe)
        view.repositoryView.buttonBox.removeButton.clicked.connect(controller._removeSelectedProbe)

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
            self._view.repositoryView,
            'Open Probe',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openProbe(filePath, nameFilter)

    def _saveSelectedProbe(self) -> None:
        current = self._view.repositoryView.treeView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view.repositoryView,
                'Save Probe',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.internalPointer().getName()
                self._repositoryPresenter.saveProbe(name, filePath, nameFilter)
        else:
            logger.error('No items are selected!')

    def _editSelectedProbe(self) -> None:
        current = self._view.repositoryView.treeView.currentIndex()

        if current.isValid():
            itemPresenter = current.internalPointer().presenter  # TODO do this cleaner
            item = itemPresenter.item
            initializerName = item.getInitializerSimpleName()

            if initializerName == 'Disk':
                diskController = DiskProbeViewController.createInstance(itemPresenter, self._view)
                diskController.openDialog()
            elif initializerName == 'FresnelZonePlate':
                fzpController = FresnelZonePlateProbeViewController.createInstance(
                    itemPresenter, self._view)
                fzpController.openDialog()
            elif initializerName == 'SuperGaussian':
                sgController = SuperGaussianProbeViewController.createInstance(
                    itemPresenter, self._view)
                sgController.openDialog()
            else:
                # FIXME FromFile
                logger.error('Unknown repository item!')
        else:
            logger.error('No items are selected!')

    def _removeSelectedProbe(self) -> None:
        current = self._view.repositoryView.treeView.currentIndex()

        if current.isValid():
            name = current.internalPointer().getName()
            self._repositoryPresenter.removeProbe(name)
        else:
            logger.error('No items are selected!')

    def _updateView(self) -> None:
        selectionModel = self._view.repositoryView.treeView.selectionModel()
        hasSelection = selectionModel.hasSelection()

        self._view.repositoryView.buttonBox.saveButton.setEnabled(hasSelection)
        self._view.repositoryView.buttonBox.editButton.setEnabled(hasSelection)
        self._view.repositoryView.buttonBox.removeButton.setEnabled(hasSelection)

        for index in selectionModel.selectedIndexes():
            node = index.internalPointer()
            self._imagePresenter.setArray(node.getArray())
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
