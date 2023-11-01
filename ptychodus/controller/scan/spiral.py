from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.scan import ScanRepositoryItemPresenter, SpiralScanInitializer
from ...view.scan import SpiralScanView, ScanEditorDialog
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class SpiralScanController(Observer):

    def __init__(self, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = SpiralScanView.createInstance()
        self._dialog = ScanEditorDialog.createInstance(presenter.name, self._view, parent)
        self._transformController = ScanTransformController.createInstance(
            presenter.item, self._dialog.transformView)
        self._initializer: SpiralScanInitializer | None = None

    @classmethod
    def editParameters(cls, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        controller._dialog.open()
        presenter.item.removeObserver(controller)

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, SpiralScanInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.numberOfPointsSpinBox.valueChanged.connect(self._initializer.setNumberOfPoints)
        self._view.radiusScalarWidget.lengthChanged.connect(
            self._initializer.setRadiusScalarInMeters)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.numberOfPointsSpinBox.setValue(self._initializer.getNumberOfPoints())
            self._view.radiusScalarWidget.setLengthInMeters(
                self._initializer.getRadiusScalarInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
