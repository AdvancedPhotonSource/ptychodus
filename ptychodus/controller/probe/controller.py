from __future__ import annotations
from typing import Callable

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.probe import (FileProbeInitializer, FresnelZonePlateProbeInitializer, ProbePresenter,
                            SuperGaussianProbeInitializer)
from ...view import (FresnelZonePlateProbeDialog, ImageView, ProbeParametersView, ProbeView,
                     ProgressBarItemDelegate, SuperGaussianProbeDialog)
from ..data import FileDialogFactory
from ..image import ImageController
from .fzp import FresnelZonePlateProbeController
from .superGaussian import SuperGaussianProbeController
from .treeModel import ProbeTreeModel, ProbeTreeNode


class ProbeParametersController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ProbePresenter,
                       view: ProbeParametersView) -> ProbeParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.sizeSpinBox.valueChanged.connect(presenter.setProbeSize)
        view.sizeSpinBox.autoToggled.connect(presenter.setAutomaticProbeSizeEnabled)
        view.energyWidget.energyChanged.connect(presenter.setProbeEnergyInElectronVolts)
        view.wavelengthWidget.setReadOnly(True)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.sizeSpinBox.setAutomatic(self._presenter.isAutomaticProbeSizeEnabled())
        self._view.sizeSpinBox.setValueAndRange(self._presenter.getProbeSize(),
                                                self._presenter.getProbeMinSize(),
                                                self._presenter.getProbeMaxSize())
        self._view.energyWidget.setEnergyInElectronVolts(
            self._presenter.getProbeEnergyInElectronVolts())
        self._view.wavelengthWidget.setLengthInMeters(self._presenter.getProbeWavelengthInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ProbeController(Observer):

    def __init__(self, presenter: ProbePresenter, view: ProbeView, imagePresenter: ImagePresenter,
                 imageView: ImageView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._imagePresenter = imagePresenter
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._parametersController = ProbeParametersController.createInstance(
            presenter, view.parametersView)
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._treeModel = ProbeTreeModel()

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, view: ProbeView,
                       imagePresenter: ImagePresenter, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeController:
        controller = cls(presenter, view, imagePresenter, imageView, fileDialogFactory)
        presenter.addObserver(controller)

        delegate = ProgressBarItemDelegate(view.modesView.treeView)
        view.modesView.treeView.setItemDelegateForColumn(1, delegate)
        view.modesView.treeView.setModel(controller._treeModel)
        view.modesView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.modesView.treeView.selectionModel().currentChanged.connect(
            controller._displayProbeMode)

        imageView.imageRibbon.indexGroupBox.setVisible(False)

        for name in presenter.getInitializerNameList():
            initAction = view.modesView.buttonBox.initializeMenu.addAction(name)
            initAction.triggered.connect(controller._createInitializerLambda(name))

        view.modesView.buttonBox.saveButton.clicked.connect(controller._saveProbe)

        controller._syncModelToView()

        return controller

    def _editProbe(self) -> None:
        initializerName = self._presenter.getInitializerName()
        initializer = self._presenter.getInitializer()

        # TODO show live update while editing
        if isinstance(initializer, FileProbeInitializer):
            filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
                self._view,
                'Open Probe',
                nameFilters=initializer.getOpenFileFilterList(),
                selectedNameFilter=initializer.getOpenFileFilter())

            if filePath:
                initializer.setOpenFilePath(filePath)
                initializer.setOpenFileFilter(nameFilter)
                self._presenter.initializeProbe()
        elif isinstance(initializer, SuperGaussianProbeInitializer):
            sgDialog = SuperGaussianProbeDialog.createInstance(self._view)
            sgDialog.setWindowTitle(initializerName)
            sgDialog.finished.connect(self._finishInitialization)
            sgController = SuperGaussianProbeController.createInstance(
                initializer, sgDialog.editorView)
            sgDialog.open()
        elif isinstance(initializer, FresnelZonePlateProbeInitializer):
            fzpDialog = FresnelZonePlateProbeDialog.createInstance(self._view)
            fzpDialog.setWindowTitle(initializerName)
            fzpDialog.finished.connect(self._finishInitialization)
            fzpController = FresnelZonePlateProbeController.createInstance(
                initializer, fzpDialog.editorView)
            fzpDialog.open()
        else:
            self._finishInitialization(QDialog.Accepted)

    def _startInitialization(self, name: str) -> None:
        self._presenter.setInitializerByName(name)
        self._editProbe()

    def _finishInitialization(self, result: int) -> None:
        if result == QDialog.Accepted:
            self._presenter.initializeProbe()

    def _createInitializerLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._startInitialization(name)

    def _renderImageData(self, index: int) -> None:
        # FIXME crashes if probe index exceeds num probe modes; clear if no selection
        array = self._presenter.getProbeMode(index)
        self._imagePresenter.setArray(array)

    def _displayProbeMode(self, current: QModelIndex, previous: QModelIndex) -> None:
        self._renderImageData(current.row())

    def _saveProbe(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Probe',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            self._presenter.saveProbe(filePath, nameFilter)

    def _syncModelToView(self) -> None:
        rootNode = ProbeTreeNode.createRoot()
        probeNode = rootNode.createChild('Current Probe', -1)

        for index in range(self._presenter.getNumberOfProbeModes()):
            power = self._presenter.getProbeModeRelativePower(index)
            powerPct = int((100 * power).to_integral_value())
            probeNode.createChild(f'Mode {index+1}', powerPct)

        self._treeModel.setRootNode(rootNode)

        current = self._view.modesView.treeView.currentIndex()
        self._renderImageData(current.row())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
