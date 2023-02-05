from __future__ import annotations
from decimal import Decimal
from pathlib import Path

from ...api.observer import Observable, Observer
from ...model.ptychonn import PtychoNNTrainingPresenter
from ...view import PtychoNNTrainingParametersView
from ..data import FileDialogFactory


class PtychoNNTrainingParametersController(Observer):

    def __init__(self, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(
            cls, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingParametersView,
            fileDialogFactory: FileDialogFactory) -> PtychoNNTrainingParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.validationSetFractionalSizeSlider.valueChanged.connect(
            presenter.setValidationSetFractionalSize)
        view.optimizationEpochsPerHalfCycleSpinBox.valueChanged.connect(
            presenter.setOptimizationEpochsPerHalfCycle)
        view.maximumLearningRateLineEdit.valueChanged.connect(presenter.setMaximumLearningRate)
        view.minimumLearningRateLineEdit.valueChanged.connect(presenter.setMinimumLearningRate)
        view.trainingEpochsSpinBox.valueChanged.connect(presenter.setTrainingEpochs)
        view.statusIntervalSpinBox.valueChanged.connect(presenter.setStatusIntervalInEpochs)
        # FIXME outputParametersView = PtychoNNOutputParametersView.createInstance()
        view.trainButton.clicked.connect(controller._train)

        controller._syncModelToView()

        return controller

    def _train(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Training Data',
            nameFilters=self._presenter.getTrainingFileFilterList(),
            selectedNameFilter=self._presenter.getTrainingFileFilter())

        if filePath:
            self._presenter.train(filePath)

    def _syncModelToView(self) -> None:
        self._view.validationSetFractionalSizeSlider.setValueAndRange(
            self._presenter.getValidationSetFractionalSize(),
            self._presenter.getValidationSetFractionalSizeLimits(),
            blockValueChangedSignal=True)

        self._view.optimizationEpochsPerHalfCycleSpinBox.blockSignals(True)
        self._view.optimizationEpochsPerHalfCycleSpinBox.setRange(
            self._presenter.getOptimizationEpochsPerHalfCycleLimits().lower,
            self._presenter.getOptimizationEpochsPerHalfCycleLimits().upper)
        self._view.optimizationEpochsPerHalfCycleSpinBox.setValue(
            self._presenter.getOptimizationEpochsPerHalfCycle())
        self._view.optimizationEpochsPerHalfCycleSpinBox.blockSignals(False)

        self._view.maximumLearningRateLineEdit.setMinimum(Decimal())
        self._view.maximumLearningRateLineEdit.setValue(self._presenter.getMaximumLearningRate())

        self._view.minimumLearningRateLineEdit.setMinimum(Decimal())
        self._view.minimumLearningRateLineEdit.setValue(self._presenter.getMinimumLearningRate())

        self._view.trainingEpochsSpinBox.blockSignals(True)
        self._view.trainingEpochsSpinBox.setRange(self._presenter.getTrainingEpochsLimits().lower,
                                                  self._presenter.getTrainingEpochsLimits().upper)
        self._view.trainingEpochsSpinBox.setValue(self._presenter.getTrainingEpochs())
        self._view.trainingEpochsSpinBox.blockSignals(False)

        self._view.statusIntervalSpinBox.blockSignals(True)
        self._view.statusIntervalSpinBox.setRange(
            self._presenter.getStatusIntervalInEpochsLimits().lower,
            self._presenter.getStatusIntervalInEpochsLimits().upper)
        self._view.statusIntervalSpinBox.setValue(self._presenter.getStatusIntervalInEpochs())
        self._view.statusIntervalSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
