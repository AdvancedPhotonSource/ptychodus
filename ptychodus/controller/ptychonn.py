from __future__ import annotations
from pathlib import Path

from PyQt5.QtWidgets import QWidget

from ..api.observer import Observable, Observer
from ..model.ptychonn import PtychoNNReconstructorLibrary, PtychoNNPresenter
from ..view import PtychoNNParametersView, PtychoNNBasicParametersView
from .data import FileDialogFactory
from .reconstructor import ReconstructorViewControllerFactory


class PtychoNNBasicParametersController(Observer):

    def __init__(self, presenter: PtychoNNPresenter, view: PtychoNNBasicParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: PtychoNNPresenter, view: PtychoNNBasicParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNBasicParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.modelStateLineEdit.editingFinished.connect(controller._syncModelStateFilePathToModel)
        view.modelStateBrowseButton.clicked.connect(controller._openModelState)
        view.numberOfConvolutionChannelsSpinBox.valueChanged.connect(
            presenter.setNumberOfConvolutionChannels)
        view.batchSizeSpinBox.valueChanged.connect(presenter.setBatchSize)
        view.useBatchNormalizationCheckBox.toggled.connect(presenter.setBatchNormalizationEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelStateFilePathToModel(self) -> None:
        self._presenter.setModelStateFilePath(Path(self._view.modelStateLineEdit.text()))

    def _openModelState(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Model State',
            nameFilters=self._presenter.getModelStateFileFilterList(),
            selectedNameFilter=self._presenter.getModelStateFileFilter())

        if filePath:
            self._presenter.setModelStateFilePath(filePath)

    def _syncModelToView(self) -> None:
        modelStateFilePath = self._presenter.getModelStateFilePath()

        if modelStateFilePath:
            self._view.modelStateLineEdit.setText(str(modelStateFilePath))
        else:
            self._view.modelStateLineEdit.clear()

        self._view.numberOfConvolutionChannelsSpinBox.blockSignals(True)
        self._view.numberOfConvolutionChannelsSpinBox.setRange(
            self._presenter.getNumberOfConvolutionChannelsLimits().lower,
            self._presenter.getNumberOfConvolutionChannelsLimits().upper)
        self._view.numberOfConvolutionChannelsSpinBox.setValue(
            self._presenter.getNumberOfConvolutionChannels())
        self._view.numberOfConvolutionChannelsSpinBox.blockSignals(False)

        self._view.batchSizeSpinBox.blockSignals(True)
        self._view.batchSizeSpinBox.setRange(self._presenter.getBatchSizeLimits().lower,
                                             self._presenter.getBatchSizeLimits().upper)
        self._view.batchSizeSpinBox.setValue(self._presenter.getBatchSize())
        self._view.batchSizeSpinBox.blockSignals(False)

        self._view.useBatchNormalizationCheckBox.setChecked(
            self._presenter.isBatchNormalizationEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PtychoNNParametersController:

    def __init__(self, presenter: PtychoNNPresenter, view: PtychoNNParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._basicParametersController = PtychoNNBasicParametersController.createInstance(
            presenter, view.basicParametersView, fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: PtychoNNPresenter, view: PtychoNNParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNParametersController:
        return cls(presenter, view, fileDialogFactory)


class PtychoNNViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: PtychoNNReconstructorLibrary,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._model = model
        self._fileDialogFactory = fileDialogFactory
        self._controllerList: list[PtychoNNParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoNN'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoNNParametersView.createInstance()

        controller = PtychoNNParametersController.createInstance(self._model.presenter, view,
                                                                 self._fileDialogFactory)
        self._controllerList.append(controller)

        return view
