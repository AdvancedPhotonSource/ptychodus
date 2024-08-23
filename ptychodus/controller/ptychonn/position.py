from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer

from ...model.ptychonn import PtychoNNPositionPredictionPresenter
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
        
        view.centralCropLineEdit.textChanged.connect(presenter.setCentralCrop)
        view.methodDropDown.currentTextChanged.connect(presenter.setMethod)
        view.numberNeighborsCollectiveSpinbox.valueChanged.connect(
            presenter.setNumberNeighborsCollective)
        view.offsetEstimatorOrderLineEdit.textChanged.connect(presenter.setOffsetEstimatorOrder)
        view.offsetEstimatorBetaLineEdit.valueChanged.connect(presenter.setOffsetEstimatorBeta)
        view.smoothConstraintWeightLineEdit.valueChanged.connect(presenter.setSmoothConstraintWeight)
        view.rectangularCheckBox.toggled.connect(presenter.setRectangularGrid)
        view.randomSeedLineEdit.textChanged.connect(presenter.setRandomSeed)
        view.debugCheckBox.toggled.connect(presenter.setDebug)
        view.registrationParametersView.registrationMethodDropDown.currentTextChanged.connect(
            presenter.setRegistrationMethod
        )
        view.registrationParametersView.hybridRegistrationTolsLineEdit.textChanged.connect(
            presenter.setHybridRegistrationTols
        )
        view.registrationParametersView.nonhybridRegistrationTolsLineEdit.textChanged.connect(
            presenter.setNonHybridRegistrationTol
        )
        view.registrationParametersView.maxShiftLineEdit.textChanged.connect(presenter.setMaxShift)
        
        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.centralCropLineEdit.setText(str(self._presenter.getCentralCrop()))
        self._view.methodDropDown.setCurrentText(str(self._presenter.getMethod()))
        
        self._view.numberNeighborsCollectiveSpinbox.blockSignals(True)
        self._view.numberNeighborsCollectiveSpinbox.setRange(
            self._presenter.getNumberNeighborsCollectiveLimits().lower,
            self._presenter.getNumberNeighborsCollectiveLimits().upper)
        self._view.numberNeighborsCollectiveSpinbox.setValue(self._presenter.getNumberNeighborsCollective())
        self._view.numberNeighborsCollectiveSpinbox.blockSignals(False)
        
        self._view.offsetEstimatorOrderLineEdit.setText(str(self._presenter.getOffsetEstimatorOrder()))
        self._view.offsetEstimatorBetaLineEdit.setValue(self._presenter.getOffsetEstimatorBeta())
        self._view.smoothConstraintWeightLineEdit.setValue(self._presenter.getSmoothConstraintWeight())
        self._view.rectangularCheckBox.setChecked(self._presenter.getRectangularGrid())
        self._view.randomSeedLineEdit.setText(str(self._presenter.getRandomSeed()))
        self._view.debugCheckBox.setChecked(self._presenter.getDebug())

        self._view.registrationParametersView.registrationMethodDropDown.setCurrentText(self._presenter.getRegistrationMethod())
        self._view.registrationParametersView.hybridRegistrationTolsLineEdit.setText(str(self._presenter.getHybridRegistrationTols()))
        self._view.registrationParametersView.nonhybridRegistrationTolsLineEdit.setText(str(self._presenter.getNonHybridRegistrationTol()))
        self._view.registrationParametersView.maxShiftLineEdit.setText(str(self._presenter.getMaxShift()))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
