from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

from ...api.observer import Observable, Observer
from ptychodus.model.ptychopinn.core import PtychoPINNTrainingPresenter
from ...view.ptychopinn import PtychoPINNOutputParametersView, PtychoPINNTrainingParametersView
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class PtychoPINNOutputParametersController(Observer):

    def __init__(self, presenter: PtychoPINNTrainingPresenter,
                 view: PtychoPINNOutputParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(
            cls, presenter: PtychoPINNTrainingPresenter, view: PtychoPINNOutputParametersView,
            fileDialogFactory: FileDialogFactory) -> PtychoPINNOutputParametersController:
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


class PtychoPINNTrainingParametersController(Observer):

    def __init__(self, presenter: PtychoPINNTrainingPresenter,
                 view: PtychoPINNTrainingParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._outputParametersController = PtychoPINNOutputParametersController.createInstance(
            presenter, view.outputParametersView, fileDialogFactory)

    @classmethod
    def createInstance(
            cls, presenter: PtychoPINNTrainingPresenter, view: PtychoPINNTrainingParametersView,
            fileDialogFactory: FileDialogFactory) -> PtychoPINNTrainingParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.validationSetFractionalSizeSlider.valueChanged.connect(
            presenter.setValidationSetFractionalSize)
        view.maximumLearningRateLineEdit.valueChanged.connect(presenter.setMaximumLearningRate)
        view.minimumLearningRateLineEdit.valueChanged.connect(presenter.setMinimumLearningRate)
        view.trainingEpochsSpinBox.valueChanged.connect(presenter.setTrainingEpochs)
        view.maeWeightLineEdit.valueChanged.connect(presenter.setMaeWeight)
        view.nllWeightLineEdit.valueChanged.connect(presenter.setNllWeight)
        view.realspaceMAEWeightLineEdit.valueChanged.connect(presenter.setRealspaceMAEWeight)
        view.realspaceWeightLineEdit.valueChanged.connect(presenter.setRealspaceWeight)

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

        self._view.maeWeightLineEdit.setValue(self._presenter.getMaeWeight())
        self._view.nllWeightLineEdit.setValue(self._presenter.getNllWeight())
        self._view.realspaceMAEWeightLineEdit.setValue(self._presenter.getRealspaceMAEWeight())
        self._view.realspaceWeightLineEdit.setValue(self._presenter.getRealspaceWeight())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
