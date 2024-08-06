from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer

from ...model.ptychonn import PtychoNNPositionPredictionPresenter
from ...model.ptychonn.position import PositionPredictionWorker
from ...view.ptychonn import PtychoNNPositionPredictionParametersView
from ..data import FileDialogFactory


class PtychoNNPositionPredictionParametersController(Observer):

    def __init__(self, presenter: PtychoNNPositionPredictionPresenter, view: PtychoNNPositionPredictionParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: PtychoNNPositionPredictionPresenter, view: PtychoNNPositionPredictionParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNPositionPredictionParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.reconstructorImagePathLineEdit.editingFinished.connect(controller._syncReconstructorImagePathToModel)
        view.reconstructorImagePathBrowseButton.clicked.connect(controller._openReconstructorImagePath)
        view.numberNeighborsCollectiveSpinbox.valueChanged.connect(
            presenter.setNumberNeighborsCollective)
        view.runButton.clicked.connect(presenter.runPositionPrediction)
#        view.numberOfConvolutionKernelsSpinBox.valueChanged.connect(
#            presenter.setNumberOfConvolutionKernels)
#        view.batchSizeSpinBox.valueChanged.connect(presenter.setBatchSize)
#        view.useBatchNormalizationCheckBox.toggled.connect(presenter.setBatchNormalizationEnabled)

        controller._syncModelToView()

        return controller

    def _syncReconstructorImagePathToModel(self) -> None:
        self._presenter.setReconstructorImageFilePath(Path(self._view.reconstructorImagePathLineEdit.text()))

    def _openReconstructorImagePath(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Reconstructor Image Path',
            nameFilters=self._presenter.getReconstructorImageFileFilterList(),
            selectedNameFilter=self._presenter.getReconstructorImageFileFilterList())

        if filePath:
            self._presenter.setReconstructorImageFilePath(filePath)

    def _syncModelToView(self) -> None:
        modelStateFilePath = self._presenter.getReconstructorImageFilePath()

        if modelStateFilePath:
            self._view.reconstructorImagePathLineEdit.setText(str(modelStateFilePath))
        else:
            self._view.reconstructorImagePathLineEdit.clear()

        self._view.numberNeighborsCollectiveSpinbox.blockSignals(True)
        self._view.numberNeighborsCollectiveSpinbox.setRange(
             self._presenter.getNumberNeighborsCollectiveLimits().lower,
             self._presenter.getNumberNeighborsCollectiveLimits().upper)
        
        # self._view.numberOfConvolutionKernelsSpinBox.blockSignals(True)
        # self._view.numberOfConvolutionKernelsSpinBox.setRange(
        #     self._presenter.getNumberOfConvolutionKernelsLimits().lower,
        #     self._presenter.getNumberOfConvolutionKernelsLimits().upper)

        # self._view.numberOfConvolutionKernelsSpinBox.setValue(
        #     self._presenter.getNumberOfConvolutionKernels())
        # self._view.numberOfConvolutionKernelsSpinBox.blockSignals(False)

        # self._view.batchSizeSpinBox.blockSignals(True)
        # self._view.batchSizeSpinBox.setRange(self._presenter.getBatchSizeLimits().lower,
        #                                      self._presenter.getBatchSizeLimits().upper)
        # self._view.batchSizeSpinBox.setValue(self._presenter.getBatchSize())
        # self._view.batchSizeSpinBox.blockSignals(False)

        # self._view.useBatchNormalizationCheckBox.setChecked(
        #     self._presenter.isBatchNormalizationEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
