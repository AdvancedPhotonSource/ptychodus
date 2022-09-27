from PyQt5.QtWidgets import QDialog, QListView

from ...api.observer import Observable, Observer
from ...model import MetadataPresenter
from ...view import DataNavigationPage, MetadataView


class MetadataController:

    def __init__(self, metadataPresenter: MetadataPresenter,
                 view: DataNavigationPage[MetadataView]) -> None:
        self._metadataPresenter = metadataPresenter
        self._view = view

    @classmethod
    def createInstance(cls, metadataPresenter: MetadataPresenter,
                       view: DataNavigationPage[MetadataView]):
        controller = cls(metadataPresenter, view)
        # TODO only show available metadata
        view.forwardButton.clicked.connect(controller._importMetadata)
        return controller

    def _importMetadata(self) -> None:
        if self._view.contentsView.detectorPixelCountCheckBox.isChecked():
            self._metadataPresenter.syncDetectorPixelCount()

        if self._view.contentsView.detectorPixelSizeCheckBox.isChecked():
            self._metadataPresenter.syncDetectorPixelSize()

        if self._view.contentsView.detectorDistanceCheckBox.isChecked():
            self._metadataPresenter.syncDetectorDistance()

        self._metadataPresenter.syncImageCrop(
            syncCenter=self._view.contentsView.imageCropCenterCheckBox.isChecked(),
            syncExtent=self._view.contentsView.imageCropExtentCheckBox.isChecked())

        if self._view.contentsView.probeEnergyCheckBox.isChecked():
            self._metadataPresenter.syncProbeEnergy()
