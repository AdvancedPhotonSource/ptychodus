from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtCore import QItemSelection
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.probe import (ApparatusPresenter, ProbeRepositoryItemPresenter,
                            ProbeRepositoryPresenter)
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

        delegate = ProgressBarItemDelegate(view.modesView.treeView)
        view.modesView.treeView.setItemDelegateForColumn(1, delegate)
        view.modesView.treeView.setModel(controller._treeModel)
        view.modesView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.modesView.treeView.selectionModel().selectionChanged.connect(
            controller._updateImageView)

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.modesView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.modesView.buttonBox.editButton.clicked.connect(controller._editSelectedProbe)
        view.modesView.buttonBox.saveButton.clicked.connect(controller._saveSelectedProbe)
        view.modesView.buttonBox.removeButton.clicked.connect(controller._removeSelectedProbe)
        imageView.imageRibbon.indexGroupBox.setVisible(False)

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
            self._view.modesView,
            'Open Probe',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openProbe(filePath, nameFilter)

    def _saveSelectedProbe(self) -> None:
        current = self._view.modesView.treeView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view.modesView,
                'Save Probe',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.sibling(current.row(), 0).data()
                self._repositoryPresenter.saveProbe(name, filePath, nameFilter)
        else:
            logger.error('No items are selected!')

    def _getSelectedItemPresenter(self) -> ProbeRepositoryItemPresenter | None:
        itemPresenter: ProbeRepositoryItemPresenter | None = None
        proxyIndex = self._view.modesView.treeView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]

        return itemPresenter

    def _editSelectedProbe(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()

        # TODO update while editing
        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
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
                logger.error('Unknown probe repository item!')

    def _removeSelectedProbe(self) -> None:
        current = self._view.modesView.treeView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeProbe(name)
        else:
            logger.error('No items are selected!')

    def _updateView(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        for index in deselected.indexes():
            self._view.modesView.buttonBox.saveButton.setEnabled(False)
            self._view.modesView.buttonBox.editButton.setEnabled(False)
            self._view.modesView.buttonBox.removeButton.setEnabled(False)
            self._imagePresenter.clearArray()
            break

        for index in selected.indexes():
            self._view.modesView.buttonBox.saveButton.setEnabled(True)
            self._view.modesView.buttonBox.editButton.setEnabled(True)
            self._view.modesView.buttonBox.removeButton.setEnabled(True)

            sourceIndex = self._proxyModel.mapToSource(index)
            itemPresenter = self._repositoryPresenter[sourceIndex.row()]
            array = itemPresenter.item.getProbeMode(index.row())  # FIXME
            self._imagePresenter.setArray(array)
            break

    def _syncModelToView(self) -> None:
        rootNode = ProbeTreeNode.createRoot()
        probeNode = rootNode.createChild('Current Probe', -1)

        for index in range(self._presenter.getNumberOfProbeModes()):
            power = self._presenter.getProbeModeRelativePower(index)
            powerPct = int((100 * power).to_integral_value())
            probeNode.createChild(f'Mode {index+1}', powerPct)

        self._treeModel.setRootNode(rootNode)

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
