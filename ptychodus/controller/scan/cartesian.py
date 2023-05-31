from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.scan import CartesianScanInitializer, ScanRepositoryItemPresenter
from ...view.scan import CartesianScanView, ScanEditorDialog
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class CartesianScanController(Observer):

    def __init__(self, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = CartesianScanView.createInstance()
        self._dialog = ScanEditorDialog.createInstance(self._view, parent)
        self._dialog.setWindowTitle(presenter.name)
        self._transformController = ScanTransformController.createInstance(
            presenter.item, self._dialog.transformView)
        self._initializer: CartesianScanInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ScanRepositoryItemPresenter,
                       parent: QWidget) -> CartesianScanController:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        return controller

    def openDialog(self) -> None:
        self._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, CartesianScanInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.numberOfPointsXSpinBox.valueChanged.connect(
            self._initializer.setNumberOfPointsX)
        self._view.numberOfPointsYSpinBox.valueChanged.connect(
            self._initializer.setNumberOfPointsY)

        self._view.stepSizeXWidget.lengthChanged.connect(self._initializer.setStepSizeXInMeters)
        self._view.stepSizeYWidget.lengthChanged.connect(self._initializer.setStepSizeYInMeters)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.numberOfPointsXSpinBox.setValue(self._initializer.getNumberOfPointsX())
            self._view.numberOfPointsYSpinBox.setValue(self._initializer.getNumberOfPointsY())

            self._view.stepSizeXWidget.setLengthInMeters(self._initializer.getStepSizeXInMeters())
            self._view.stepSizeYWidget.setLengthInMeters(self._initializer.getStepSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
