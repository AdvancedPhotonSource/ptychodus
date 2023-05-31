from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.probe import DiskProbeInitializer, ProbeRepositoryItemPresenter
from ...view.probe import DiskProbeView, ProbeEditorDialog

logger = logging.getLogger(__name__)


class DiskProbeViewController(Observer):

    def __init__(self, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = DiskProbeView.createInstance()
        self._dialog = ProbeEditorDialog.createInstance(self._view, parent)
        self._initializer: DiskProbeInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ProbeRepositoryItemPresenter,
                       parent: QWidget) -> DiskProbeViewController:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        return controller

    def openDialog(self) -> None:
        self._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, DiskProbeInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.diameterWidget.lengthChanged.connect(initializer.setDiameterInMeters)
        self._view.numberOfModesSpinBox.valueChanged.connect(self._item.setNumberOfModes)
        self._view.testPatternCheckBox.toggled.connect(initializer.setTestPattern)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.diameterWidget.setLengthInMeters(self._initializer.getDiameterInMeters())

            self._view.numberOfModesSpinBox.blockSignals(True)
            self._view.numberOfModesSpinBox.setRange(self._item.getNumberOfModesLimits().lower,
                                                     self._item.getNumberOfModesLimits().upper)
            self._view.numberOfModesSpinBox.setValue(self._item.getNumberOfModes())
            self._view.numberOfModesSpinBox.blockSignals(False)

            self._view.testPatternCheckBox.setChecked(self._initializer.isTestPattern())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()
