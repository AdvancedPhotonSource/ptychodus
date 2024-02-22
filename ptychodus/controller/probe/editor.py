from __future__ import annotations
import logging

from PyQt5.QtWidgets import QButtonGroup

from ...api.observer import Observable, Observer
from ...model.probe import ProbeModeDecayType, ProbeRepositoryItem
from ...view.probe import ProbeModesView

logger = logging.getLogger(__name__)


class ProbeModesController(Observer):

    def __init__(self, item: ProbeRepositoryItem, view: ProbeModesView) -> None:
        super().__init__()
        self._item = item
        self._view = view
        self._decayTypeButtonGroup = QButtonGroup()

    @classmethod
    def createInstance(cls, item: ProbeRepositoryItem,
                       view: ProbeModesView) -> ProbeModesController:
        controller = cls(item, view)
        controller._decayTypeButtonGroup.addButton(view.polynomialDecayButton)
        controller._decayTypeButtonGroup.addButton(view.exponentialDecayButton)
        controller._syncModelToView()

        view.numberOfModesSpinBox.valueChanged.connect(item.setNumberOfModes)
        view.orthogonalizeModesCheckBox.toggled.connect(item.setOrthogonalizeModesEnabled)
        view.exponentialDecayButton.toggled.connect(controller._exponentialDecayButtonToggled)
        view.decayRatioSlider.valueChanged.connect(item.setModeDecayRatio)
        item.addObserver(controller)
        return controller

    def _exponentialDecayButtonToggled(self, checked: bool) -> None:
        decayType = ProbeModeDecayType.EXPONENTIAL if checked else ProbeModeDecayType.POLYNOMIAL
        self._item.setModeDecayType(decayType)

    def _syncDecayTypeModelToView(self) -> None:
        if self._item.getModeDecayType() == ProbeModeDecayType.EXPONENTIAL:
            self._view.exponentialDecayButton.setChecked(True)
        else:
            self._view.polynomialDecayButton.setChecked(True)

    def _syncModelToView(self) -> None:
        self._view.numberOfModesSpinBox.blockSignals(True)
        self._view.numberOfModesSpinBox.setRange(self._item.getNumberOfModesLimits().lower,
                                                 self._item.getNumberOfModesLimits().upper)
        self._view.numberOfModesSpinBox.setValue(self._item.getNumberOfModes())
        self._view.numberOfModesSpinBox.blockSignals(False)

        self._view.orthogonalizeModesCheckBox.setChecked(self._item.isOrthogonalizeModesEnabled)
        self._syncDecayTypeModelToView()
        self._view.decayRatioSlider.setValueAndRange(self._item.getModeDecayRatio(),
                                                     self._item.getModeDecayRatioLimits())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
