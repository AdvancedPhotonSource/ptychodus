from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
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
        view.numberOfConvolutionChannelsSpinBox.valueChanged.connect(
            presenter.setNumberOfConvolutionChannels)
        view.batchSizeSpinBox.valueChanged.connect(presenter.setBatchSize)
        view.useBatchNormalizationCheckBox.toggled.connect(presenter.setBatchNormalizationEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelStateFilePathToModel(self) -> None:
        self._presenter.setStateFilePath(Path(self._view.modelStateLineEdit.text()))

    def _openModelState(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Model State',
            nameFilters=self._presenter.getStateFileFilterList(),
            selectedNameFilter=self._presenter.getStateFileFilter())

        if filePath:
            self._presenter.setStateFilePath(filePath)

    def _syncModelToView(self) -> None:
        modelStateFilePath = self._presenter.getStateFilePath()

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
