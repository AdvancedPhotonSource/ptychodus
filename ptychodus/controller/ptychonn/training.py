from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.ptychonn import PtychoNNTrainingPresenter
from ...view.ptychonn import PtychoNNTrainingArtifactsView, PtychoNNTrainingParametersView
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class PtychoNNTrainingArtifactsController(Observer):

    def __init__(self, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingArtifactsView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(
            cls, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingArtifactsView,
            fileDialogFactory: FileDialogFactory) -> PtychoNNTrainingArtifactsController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setSaveTrainingArtifactsEnabled)

        view.directoryLineEdit.editingFinished.connect(
            controller._syncTrainingArtifactsDirectoryToModel)
        view.directoryBrowseButton.clicked.connect(controller._browseTrainingArtifactsDirectory)
        view.suffixLineEdit.editingFinished.connect(controller._syncTrainingArtifactsSuffixToModel)

        controller._syncModelToView()

        return controller

    def _syncTrainingArtifactsDirectoryToModel(self) -> None:
        self._presenter.setTrainingArtifactsDirectory(Path(self._view.directoryLineEdit.text()))

    def _browseTrainingArtifactsDirectory(self) -> None:
        dirPath = self._fileDialogFactory.getExistingDirectoryPath(
            self._view, 'Choose Training Artifacts Directory')

        if dirPath:
            self._presenter.setTrainingArtifactsDirectory(dirPath)

    def _syncTrainingArtifactsSuffixToModel(self) -> None:
        self._presenter.setTrainingArtifactsSuffix(self._view.suffixLineEdit.text())

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isSaveTrainingArtifactsEnabled())
        outputPath = self._presenter.getTrainingArtifactsDirectory()

        if outputPath:
            self._view.directoryLineEdit.setText(str(outputPath))
        else:
            self._view.directoryLineEdit.clear()

        outputSuffix = self._presenter.getTrainingArtifactsSuffix()

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
        self._trainingArtifactsController = PtychoNNTrainingArtifactsController.createInstance(
            presenter, view.trainingArtifactsView, fileDialogFactory)

    @classmethod
    def createInstance(
            cls, presenter: PtychoNNTrainingPresenter, view: PtychoNNTrainingParametersView,
            fileDialogFactory: FileDialogFactory) -> PtychoNNTrainingParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.validationSetFractionalSizeSlider.valueChanged.connect(
            presenter.setValidationSetFractionalSize)
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
