from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.object import ObjectRepositoryItemPresenter, RandomObjectInitializer
from ...view.object import ObjectEditorDialog, RandomObjectView

logger = logging.getLogger(__name__)


class RandomObjectViewController(Observer):

    def __init__(self, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = RandomObjectView.createInstance()
        self._dialog = ObjectEditorDialog.createInstance(presenter.name, self._view, parent)
        self._initializer: RandomObjectInitializer | None = None

    @classmethod
    def editParameters(cls, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        controller._dialog.open()
        presenter.item.removeObserver(controller)

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, RandomObjectInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.numberOfLayersSpinBox.valueChanged.connect(initializer.setNumberOfLayers)
        self._view.extraPaddingXSpinBox.valueChanged.connect(initializer.setExtraPaddingX)
        self._view.extraPaddingYSpinBox.valueChanged.connect(initializer.setExtraPaddingY)

        self._view.amplitudeMeanSlider.valueChanged.connect(initializer.setAmplitudeMean)
        self._view.amplitudeDeviationSlider.valueChanged.connect(initializer.setAmplitudeDeviation)
        self._view.phaseDeviationSlider.valueChanged.connect(initializer.setPhaseDeviation)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.numberOfLayersSpinBox.blockSignals(True)
            self._view.numberOfLayersSpinBox.setRange(
                self._initializer.getNumberOfLayersLimits().lower,
                self._initializer.getNumberOfLayersLimits().upper)
            self._view.numberOfLayersSpinBox.setValue(self._initializer.getNumberOfLayers())
            self._view.numberOfLayersSpinBox.blockSignals(False)

            self._view.extraPaddingXSpinBox.blockSignals(True)
            self._view.extraPaddingXSpinBox.setRange(
                self._initializer.getExtraPaddingXLimits().lower,
                self._initializer.getExtraPaddingXLimits().upper)
            self._view.extraPaddingXSpinBox.setValue(self._initializer.getExtraPaddingX())
            self._view.extraPaddingXSpinBox.blockSignals(False)

            self._view.extraPaddingYSpinBox.blockSignals(True)
            self._view.extraPaddingYSpinBox.setRange(
                self._initializer.getExtraPaddingYLimits().lower,
                self._initializer.getExtraPaddingYLimits().upper)
            self._view.extraPaddingYSpinBox.setValue(self._initializer.getExtraPaddingY())
            self._view.extraPaddingYSpinBox.blockSignals(False)

            self._view.amplitudeMeanSlider.setValueAndRange(
                self._initializer.getAmplitudeMean(), self._initializer.getAmplitudeMeanLimits())
            self._view.amplitudeDeviationSlider.setValueAndRange(
                self._initializer.getAmplitudeDeviation(),
                self._initializer.getAmplitudeDeviationLimits())
            self._view.phaseDeviationSlider.setValueAndRange(
                self._initializer.getPhaseDeviation(), self._initializer.getPhaseDeviationLimits())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
