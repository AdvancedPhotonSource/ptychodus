from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.probe import ProbeRepositoryItemPresenter, SuperGaussianProbeInitializer
from ...view.probe import SuperGaussianProbeView
from .editor import ProbeEditorViewController

logger = logging.getLogger(__name__)


class SuperGaussianProbeViewController(Observer):

    def __init__(self, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = SuperGaussianProbeView.createInstance()
        self._initializer: SuperGaussianProbeInitializer | None = None

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

        if isinstance(initializer, SuperGaussianProbeInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.annularRadiusWidget.lengthChanged.connect(initializer.setAnnularRadiusInMeters)
        self._view.fwhmWidget.lengthChanged.connect(initializer.setFullWidthAtHalfMaximumInMeters)
        self._view.orderParameterWidget.valueChanged.connect(initializer.setOrderParameter)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.annularRadiusWidget.setLengthInMeters(
                self._initializer.getAnnularRadiusInMeters())
            self._view.fwhmWidget.setLengthInMeters(
                self._initializer.getFullWidthAtHalfMaximumInMeters())
            self._view.orderParameterWidget.setValue(self._initializer.getOrderParameter())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
