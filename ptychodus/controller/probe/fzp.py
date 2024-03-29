from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.probe import FresnelZonePlateProbeInitializer, ProbeRepositoryItemPresenter
from ...view.probe import FresnelZonePlateProbeView
from .editor import ProbeEditorViewController

logger = logging.getLogger(__name__)


class FresnelZonePlateProbeViewController(Observer):

    def __init__(self, presenter: ProbeRepositoryItemPresenter, parent: QWidget) -> None:
        super().__init__()
        self._item = presenter.item
        self._view = FresnelZonePlateProbeView.createInstance()
        self._initializer: FresnelZonePlateProbeInitializer | None = None

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

        if isinstance(initializer, FresnelZonePlateProbeInitializer):
            self._initializer = initializer
            self._initializer.addObserver(self)
        else:
            logger.error('Null initializer!')
            return

        for presets in initializer.getPresetsList():
            self._view.presetsComboBox.addItem(presets)

        self._view.presetsComboBox.textActivated.connect(initializer.setPresets)
        self._view.zonePlateDiameterWidget.lengthChanged.connect(
            initializer.setZonePlateDiameterInMeters)
        self._view.outermostZoneWidthWidget.lengthChanged.connect(
            initializer.setOutermostZoneWidthInMeters)
        self._view.beamstopDiameterWidget.lengthChanged.connect(
            initializer.setCentralBeamstopDiameterInMeters)
        self._view.defocusDistanceWidget.lengthChanged.connect(
            initializer.setDefocusDistanceInMeters)

    def _syncModelToView(self) -> None:
        if self._initializer is None:
            logger.error('Null initializer!')
        else:
            presets = self._initializer.getPresets()
            arePresetsCustomized = (presets == FresnelZonePlateProbeInitializer.CUSTOM_PRESET)

            self._view.presetsComboBox.setCurrentText(presets)
            self._view.zonePlateDiameterWidget.setEnabled(arePresetsCustomized)
            self._view.zonePlateDiameterWidget.setLengthInMeters(
                self._initializer.getZonePlateDiameterInMeters())
            self._view.outermostZoneWidthWidget.setEnabled(arePresetsCustomized)
            self._view.outermostZoneWidthWidget.setLengthInMeters(
                self._initializer.getOutermostZoneWidthInMeters())
            self._view.beamstopDiameterWidget.setEnabled(arePresetsCustomized)
            self._view.beamstopDiameterWidget.setLengthInMeters(
                self._initializer.getCentralBeamstopDiameterInMeters())
            self._view.defocusDistanceWidget.setLengthInMeters(
                self._initializer.getDefocusDistanceInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()
        elif observable is self._initializer:
            self._syncModelToView()
