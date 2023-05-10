from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.scan import LissajousScanInitializer, ScanRepositoryItemPresenter
from ...view.scan import LissajousScanView, ScanEditorDialog
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class LissajousScanController(Observer):

    def __init__(self, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = LissajousScanView.createInstance()
        self._dialog = ScanEditorDialog.createInstance(self._view, parent)
        self._dialog.setWindowTitle(presenter.name)
        self._transformController = ScanTransformController.createInstance(
            presenter.item, self._dialog.transformView)
        self._initializer: LissajousScanInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ScanRepositoryItemPresenter,
                       parent: QWidget) -> LissajousScanController:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        return controller

    def openDialog(self) -> None:
        self._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, LissajousScanInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.numberOfPointsSpinBox.valueChanged.connect(self._initializer.setNumberOfPoints)
        self._view.amplitudeXWidget.lengthChanged.connect(self._initializer.setAmplitudeXInMeters)
        self._view.amplitudeYWidget.lengthChanged.connect(self._initializer.setAmplitudeYInMeters)
        self._view.angularStepXWidget.angleChanged.connect(
            self._initializer.setAngularStepXInTurns)
        self._view.angularStepYWidget.angleChanged.connect(
            self._initializer.setAngularStepYInTurns)
        self._view.angularShiftWidget.angleChanged.connect(
            self._initializer.setAngularShiftInTurns)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.numberOfPointsSpinBox.setValue(self._initializer.getNumberOfPoints())
            self._view.amplitudeXWidget.setLengthInMeters(
                self._initializer.getAmplitudeXInMeters())
            self._view.amplitudeYWidget.setLengthInMeters(
                self._initializer.getAmplitudeYInMeters())
            self._view.angularStepXWidget.setAngleInTurns(
                self._initializer.getAngularStepXInTurns())
            self._view.angularStepYWidget.setAngleInTurns(
                self._initializer.getAngularStepYInTurns())
            self._view.angularShiftWidget.setAngleInTurns(
                self._initializer.getAngularShiftInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
