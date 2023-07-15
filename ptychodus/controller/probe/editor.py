from __future__ import annotations
from typing import Optional
import logging

from PyQt5.QtWidgets import QButtonGroup, QGroupBox, QWidget

from ...api.observer import Observable, Observer
from ...model.probe import ProbeModeDecayType, ProbeRepositoryItem, ProbeRepositoryItemPresenter
from ...view.probe import ProbeEditorDialog, ProbeModesView

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

        self._syncDecayTypeModelToView()
        self._view.decayRatioSlider.setValueAndRange(self._item.getModeDecayRatio(),
                                                     self._item.getModeDecayRatioLimits())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()


class ProbeEditorViewController:

    def __init__(self, presenter: ProbeRepositoryItemPresenter, editorView: QGroupBox,
                 parent: Optional[QWidget]) -> None:
        self._dialog = ProbeEditorDialog.createInstance(presenter.name, editorView, parent)
        self._modesController = ProbeModesController.createInstance(presenter.item,
                                                                    self._dialog.modesView)

    @classmethod
    def editParameters(cls,
                       presenter: ProbeRepositoryItemPresenter,
                       editorView: QGroupBox,
                       parent: Optional[QWidget] = None) -> None:
        vc = cls(presenter, editorView, parent)
        vc._dialog.open()
