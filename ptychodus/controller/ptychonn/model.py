from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer

from ...model.ptychonn import PtychoNNModelPresenter
from ...view.ptychonn import PtychoNNModelParametersView
from ..data import FileDialogFactory


class PtychoNNModelParametersController(Observer):

    def __init__(self, presenter: PtychoNNModelPresenter, view: PtychoNNModelParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: PtychoNNModelPresenter, view: PtychoNNModelParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNModelParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.modelStateLineEdit.editingFinished.connect(controller._syncModelStateFilePathToModel)
        view.modelStateBrowseButton.clicked.connect(controller._openModelState)
        view.numberOfConvolutionKernelsSpinBox.valueChanged.connect(
            presenter.setNumberOfConvolutionKernels)
        view.batchSizeSpinBox.valueChanged.connect(presenter.setBatchSize)
        view.useBatchNormalizationCheckBox.toggled.connect(presenter.setBatchNormalizationEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelStateFilePathToModel(self) -> None:
        self._presenter.setModelFilePath(Path(self._view.modelStateLineEdit.text()))

    def _openModelState(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Model State',
            nameFilters=self._presenter.getSaveModelFileFilterList(),
            selectedNameFilter=self._presenter.getSaveModelFileFilter())

        if filePath:
            self._presenter.setModelFilePath(filePath)

    def _syncModelToView(self) -> None:
        modelStateFilePath = self._presenter.getModelFilePath()

        if modelStateFilePath:
            self._view.modelStateLineEdit.setText(str(modelStateFilePath))
        else:
            self._view.modelStateLineEdit.clear()

        self._view.numberOfConvolutionKernelsSpinBox.blockSignals(True)
        self._view.numberOfConvolutionKernelsSpinBox.setRange(
            self._presenter.getNumberOfConvolutionKernelsLimits().lower,
            self._presenter.getNumberOfConvolutionKernelsLimits().upper)
        self._view.numberOfConvolutionKernelsSpinBox.setValue(
            self._presenter.getNumberOfConvolutionKernels())
        self._view.numberOfConvolutionKernelsSpinBox.blockSignals(False)

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
