from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.object import ObjectRepositoryItemPresenter, RandomObjectInitializer
from ...view import ObjectEditorDialog, RandomObjectView

logger = logging.getLogger(__name__)


class RandomObjectViewController(Observer):

    def __init__(self, presenter: ObjectRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = RandomObjectView.createInstance()
        self._dialog = ObjectEditorDialog.createInstance(self._view, parent)
        self._dialog.setWindowTitle(presenter.name)
        self._initializer: RandomObjectInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ObjectRepositoryItemPresenter,
                       parent: QWidget) -> RandomObjectViewController:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        return controller

    def openDialog(self) -> None:
        self._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, RandomObjectInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.extraPaddingXSpinBox.valueChanged.connect(initializer.setExtraPaddingX)
        self._view.extraPaddingYSpinBox.valueChanged.connect(initializer.setExtraPaddingY)

        self._view.amplitudeMeanSlider.valueChanged.connect(initializer.setAmplitudeMean)
        self._view.amplitudeDeviationSlider.valueChanged.connect(initializer.setAmplitudeDeviation)

        self._view.randomizePhaseCheckBox.toggled.connect(initializer.setRandomizePhaseEnabled)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.extraPaddingXSpinBox.setValue(self._initializer.getExtraPaddingX())
            self._view.extraPaddingYSpinBox.setValue(self._initializer.getExtraPaddingY())

            self._view.amplitudeMeanSlider.setValueAndRange(
                self._initializer.getAmplitudeMean(), self._initializer.getAmplitudeMeanLimits())
            self._view.amplitudeDeviationSlider.setValueAndRange(
                self._initializer.getAmplitudeDeviation(),
                self._initializer.getAmplitudeDeviationLimits())

            self._view.randomizePhaseCheckBox.setChecked(
                self._initializer.isRandomizePhaseEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
