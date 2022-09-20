from PyQt5.QtWidgets import QDialog, QListView

from ..api.observer import Observable, Observer
from ..model import ObjectPresenter, ProbePresenter, ScanPresenter, MetadataPresenter
from ..view import MetadataDialog


class MetadataController(Observer):

    def __init__(self, probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
                 metadataPresenter: MetadataPresenter, dialog: MetadataDialog) -> None:
        super().__init__()
        self._probePresenter = probePresenter
        self._objectPresenter = objectPresenter
        self._metadataPresenter = metadataPresenter
        self._dialog = dialog

    @classmethod
    def createInstance(cls, probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
                       metadataPresenter: MetadataPresenter, dialog: MetadataDialog):
        controller = cls(probePresenter, objectPresenter, metadataPresenter, dialog)
        metadataPresenter.addObserver(controller)
        dialog.finished.connect(controller._importSettings)
        return controller

    def _importSettings(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        if self._dialog.valuesGroupBox.detectorPixelCountCheckBox.isChecked():
            self._metadataPresenter.syncDetectorPixelCount()

        if self._dialog.valuesGroupBox.detectorPixelSizeCheckBox.isChecked():
            self._metadataPresenter.syncDetectorPixelSize()

        if self._dialog.valuesGroupBox.detectorDistanceCheckBox.isChecked():
            self._metadataPresenter.syncDetectorDistance()

        self._metadataPresenter.syncImageCrop(
            syncCenter=self._dialog.valuesGroupBox.imageCropCenterCheckBox.isChecked(),
            syncExtent=self._dialog.valuesGroupBox.imageCropExtentCheckBox.isChecked())

        if self._dialog.valuesGroupBox.probeEnergyCheckBox.isChecked():
            self._metadataPresenter.syncProbeEnergy()

        if self._dialog.optionsGroupBox.loadScanCheckBox.isChecked():
            self._metadataPresenter.loadScanFile()

        if self._dialog.optionsGroupBox.reinitializeProbeCheckBox.isChecked():
            self._probePresenter.initializeProbe()

        if self._dialog.optionsGroupBox.reinitializeObjectCheckBox.isChecked():
            self._objectPresenter.initializeObject()

    def update(self, observable: Observable) -> None:
        if observable is self._metadataPresenter:
            self._dialog.open()
