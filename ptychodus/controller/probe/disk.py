from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.probe import DiskProbeInitializer, ProbeRepositoryItemPresenter
from ...view.probe import DiskProbeView
from .editor import ProbeEditorViewController

logger = logging.getLogger(__name__)


class DiskProbeViewController(Observer):

    def __init__(self, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = DiskProbeView.createInstance()
        self._initializer: DiskProbeInitializer | None = None

    @classmethod
    def editParameters(cls, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        ProbeEditorViewController.editParameters(presenter, controller._view, parent)
        presenter.item.removeObserver(controller)

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, DiskProbeInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.diameterWidget.lengthChanged.connect(initializer.setDiameterInMeters)
        self._view.testPatternCheckBox.toggled.connect(initializer.setTestPattern)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.diameterWidget.setLengthInMeters(self._initializer.getDiameterInMeters())
            self._view.testPatternCheckBox.setChecked(self._initializer.isTestPattern())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
