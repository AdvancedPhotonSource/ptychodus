from __future__ import annotations
from decimal import Decimal

from PyQt5.QtWidgets import QWidget

from ..model import Observer, Observable  # TODO , PtychoPyBackend, PtychoPyPresenter
from ..view import PtychoPyParametersView, PtychoPyBasicView, PtychoPyAdvancedView
from .reconstructor import ReconstructorViewControllerFactory


class PtychoPyBasicParametersController(Observer):
    def __init__(self, presenter: PtychoPyPresenter, view: PtychoPyBasicView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: PtychoPyPresenter,
                       view: PtychoPyBasicView) -> PtychoPyBasicParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.probeModesSpinBox.valueChanged.connect(presenter.setProbeModes)
        view.thresholdSpinBox.valueChanged.connect(presenter.setThreshold)
        view.iterationLimitSpinBox.valueChanged.connect(presenter.setReconstructionIterations)
        view.timeLimitSpinBox.valueChanged.connect(presenter.setReconstructionTimeInSeconds)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.probeModesSpinBox.blockSignals(True)
        self._view.probeModesSpinBox.setRange(self._presenter.getMinProbeModes(),
                                              self._presenter.getMaxProbeModes())
        self._view.probeModesSpinBox.setValue(self._presenter.getProbeModes())
        self._view.probeModesSpinBox.blockSignals(False)

        self._view.thresholdSpinBox.blockSignals(True)
        self._view.thresholdSpinBox.setRange(self._presenter.getMinThreshold(),
                                             self._presenter.getMaxThreshold())
        self._view.thresholdSpinBox.setValue(self._presenter.getThreshold())
        self._view.thresholdSpinBox.blockSignals(False)

        self._view.iterationLimitSpinBox.blockSignals(True)
        self._view.iterationLimitSpinBox.setRange(self._presenter.getMinReconstructionIterations(),
                                                  self._presenter.getMaxReconstructionIterations())
        self._view.iterationLimitSpinBox.setValue(self._presenter.getReconstructionIterations())
        self._view.iterationLimitSpinBox.blockSignals(False)

        self._view.timeLimitSpinBox.blockSignals(True)
        self._view.timeLimitSpinBox.setRange(self._presenter.getMinReconstructionTimeInSeconds(),
                                             self._presenter.getMaxReconstructionTimeInSeconds())
        self._view.timeLimitSpinBox.setValue(self._presenter.getReconstructionTimeInSeconds())
        self._view.timeLimitSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PtychoPyAdvancedParametersController(Observer):
    def __init__(self, presenter: PtychoPyPresenter, view: PtychoPyAdvancedView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: PtychoPyPresenter,
                       view: PtychoPyAdvancedView) -> PtychoPyAdvancedParametersController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.calculateRMSCheckBox.toggled.connect(presenter.setCalculateRMSEnabled)
        view.updateProbeSpinBox.valueChanged.connect(presenter.setUpdateProbe)
        view.updateModesSpinBox.valueChanged.connect(presenter.setUpdateModes)
        view.phaseConstraintSpinBox.valueChanged.connect(presenter.setPhaseConstraint)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.calculateRMSCheckBox.setChecked(self._presenter.isCalculateRMSEnabled())

        self._view.updateProbeSpinBox.blockSignals(True)
        self._view.updateProbeSpinBox.setRange(self._presenter.getMinUpdateProbe(),
                                               self._presenter.getMaxUpdateProbe())
        self._view.updateProbeSpinBox.setValue(self._presenter.getUpdateProbe())
        self._view.updateProbeSpinBox.blockSignals(False)

        self._view.updateModesSpinBox.blockSignals(True)
        self._view.updateModesSpinBox.setRange(self._presenter.getMinUpdateModes(),
                                               self._presenter.getMaxUpdateModes())
        self._view.updateModesSpinBox.setValue(self._presenter.getUpdateModes())
        self._view.updateModesSpinBox.blockSignals(False)

        self._view.phaseConstraintSpinBox.blockSignals(True)
        self._view.phaseConstraintSpinBox.setRange(self._presenter.getMinPhaseConstraint(),
                                                   self._presenter.getMaxPhaseConstraint())
        self._view.phaseConstraintSpinBox.setValue(self._presenter.getPhaseConstraint())
        self._view.phaseConstraintSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PtychoPyParametersController:
    def __init__(self, model: PtychoPyBackend, view: PtychoPyParametersView) -> None:
        self._model = model
        self._view = view
        self._basicParametersController = PtychoPyBasicParametersController.createInstance(
            model.presenter, view.basicView)
        self._advancedParametersController = PtychoPyAdvancedParametersController.createInstance(
            model.presenter, view.advancedView)

    @classmethod
    def createInstance(cls, model: PtychoPyBackend,
                       view: PtychoPyParametersView) -> PtychoPyParametersController:
        controller = cls(model, view)
        return controller


class PtychoPyViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtychoPyBackend) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[PtychoPyParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoPy'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoPyParametersView.createInstance()

        controller = PtychoPyParametersController.createInstance(self._model, view)
        self._controllerList.append(controller)

        return view
