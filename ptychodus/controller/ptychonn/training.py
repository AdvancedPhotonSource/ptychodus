from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

from ...api.observer import Observable, Observer
from ...model.ptychonn import PtychoNNTrainingPresenter
from ...view.ptychonn import PtychoNNOutputParametersView, PtychoNNTrainingParametersView
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class PtychoNNOutputParametersController(Observer):

    def __init__(self, presenter: PtychoNNTrainingPresenter, view: PtychoNNOutputParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, presenter: PtychoNNTrainingPresenter,
                       view: PtychoNNOutputParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNOutputParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setSaveTrainingArtifactsEnabled)

        view.pathLineEdit.editingFinished.connect(controller._syncOutputPathToModel)
        view.pathBrowseButton.clicked.connect(controller._browseOutputPath)
        view.suffixLineEdit.editingFinished.connect(controller._syncOutputSuffixToModel)

        controller._syncModelToView()

        return controller

    def _syncOutputPathToModel(self) -> None:
        self._presenter.setOutputPath(Path(self._view.pathLineEdit.text()))

    def _browseOutputPath(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(
            self._view, 'Choose Training Output Data Directory')

        if dirPath:
            self._presenter.setOutputPath(dirPath)

    def _syncOutputSuffixToModel(self) -> None:
        self._presenter.setOutputSuffix(self._view.suffixLineEdit.text())

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isSaveTrainingArtifactsEnabled())
        outputPath = self._presenter.getOutputPath()

        if outputPath:
            self._view.pathLineEdit.setText(str(outputPath))
        else:
            self._view.pathLineEdit.clear()

        outputSuffix = self._presenter.getOutputSuffix()

        if outputSuffix:
            self._view.suffixLineEdit.setText(str(outputSuffix))
        else:
            self._view.suffixLineEdit.clear()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PtychoNNTrainingParametersController(Observer):

    def __init__(self, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._outputParametersController = PtychoNNOutputParametersController.createInstance(
            presenter, view.outputParametersView, fileDialogFactory)

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

        controller._syncModelToView()

        return controller

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
