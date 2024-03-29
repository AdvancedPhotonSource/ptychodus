from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.scan import ScanRepositoryItemPresenter, ConcentricScanInitializer
from ...view.scan import ConcentricScanView, ScanEditorDialog
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class ConcentricScanController(Observer):

    def __init__(self, presenter: ScanRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = ConcentricScanView.createInstance()
        self._dialog = ScanEditorDialog.createInstance(presenter.name, self._view, parent)
        self._transformController = ScanTransformController.createInstance(
            presenter.item, self._dialog.transformView)
        self._initializer: ConcentricScanInitializer | None = None

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

        if isinstance(initializer, ConcentricScanInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.radialStepSizeWidget.lengthChanged.connect(
            self._initializer.setRadialStepSizeInMeters)
        self._view.numberOfShellsSpinBox.valueChanged.connect(self._initializer.setNumberOfShells)
        self._view.numberOfPointsInFirstShellSpinBox.valueChanged.connect(
            self._initializer.setNumberOfPointsInFirstShell)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.radialStepSizeWidget.setLengthInMeters(
                self._initializer.getRadialStepSizeInMeters())
            self._view.numberOfShellsSpinBox.setValue(self._initializer.getNumberOfShells())
            self._view.numberOfPointsInFirstShellSpinBox.setValue(
                self._initializer.getNumberOfPointsInFirstShell())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
