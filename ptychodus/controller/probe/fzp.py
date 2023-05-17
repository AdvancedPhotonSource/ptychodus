from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.probe import FresnelZonePlateProbeInitializer, ProbeRepositoryItemPresenter
from ...view.probe import FresnelZonePlateProbeView, ProbeEditorDialog

logger = logging.getLogger(__name__)


class FresnelZonePlateProbeViewController(Observer):

    def __init__(self, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = FresnelZonePlateProbeView.createInstance()
        self._dialog = ProbeEditorDialog.createInstance(self._view, parent)
        self._initializer: FresnelZonePlateProbeInitializer | None = None

    @classmethod
    def createInstance(cls, presenter: ProbeRepositoryItemPresenter,
                       parent: QWidget) -> FresnelZonePlateProbeViewController:
        controller = cls(presenter, parent)
        controller._updateInitializer()
        controller._syncModelToView()
        presenter.item.addObserver(controller)
        return controller

    def openDialog(self) -> None:
        self._dialog.open()

    def _updateInitializer(self) -> None:
        initializer = self._item.getInitializer()

        if isinstance(initializer, FresnelZonePlateProbeInitializer):
            self._initializer = initializer
        else:
            logger.error('Null initializer!')
            return

        self._view.zonePlateRadiusWidget.lengthChanged.connect(
            initializer.setZonePlateRadiusInMeters)
        self._view.outermostZoneWidthWidget.lengthChanged.connect(
            initializer.setOutermostZoneWidthInMeters)
        self._view.beamstopDiameterWidget.lengthChanged.connect(
            initializer.setCentralBeamstopDiameterInMeters)
        self._view.defocusDistanceWidget.lengthChanged.connect(
            initializer.setDefocusDistanceInMeters)
        self._view.numberOfModesSpinBox.valueChanged.connect(self._item.setNumberOfProbeModes)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            self._view.zonePlateRadiusWidget.setLengthInMeters(
                self._initializer.getZonePlateRadiusInMeters())
            self._view.outermostZoneWidthWidget.setLengthInMeters(
                self._initializer.getOutermostZoneWidthInMeters())
            self._view.beamstopDiameterWidget.setLengthInMeters(
                self._initializer.getCentralBeamstopDiameterInMeters())
            self._view.defocusDistanceWidget.setLengthInMeters(
                self._initializer.getDefocusDistanceInMeters())
            self._view.numberOfModesSpinBox.setValue(self._item.getNumberOfProbeModes())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()
