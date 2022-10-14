from PyQt5.QtWidgets import QDialog, QListView

from ...api.observer import Observable, Observer
from ...model import MetadataPresenter
from ...view import DataNavigationPage, MetadataView


class MetadataController(Observer):

    def __init__(self, presenter: MetadataPresenter,
                 view: DataNavigationPage[MetadataView]) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: MetadataPresenter, view: DataNavigationPage[MetadataView]):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.forwardButton.clicked.connect(controller._importMetadata)

        controller._syncModelToView()
        return controller

    def _importMetadata(self) -> None:
        if self._view.contentsView.detectorPixelCountCheckBox.isChecked():
            self._presenter.syncDetectorPixelCount()

        if self._view.contentsView.detectorPixelSizeCheckBox.isChecked():
            self._presenter.syncDetectorPixelSize()

        if self._view.contentsView.detectorDistanceCheckBox.isChecked():
            self._presenter.syncDetectorDistance()

        self._presenter.syncPatternCrop(
            syncCenter=self._view.contentsView.patternCropCenterCheckBox.isChecked(),
            syncExtent=self._view.contentsView.patternCropExtentCheckBox.isChecked())

        if self._view.contentsView.probeEnergyCheckBox.isChecked():
            self._presenter.syncProbeEnergy()

    def _syncModelToView(self) -> None:
        canSyncDetectorPixelCount = self._presenter.canSyncDetectorPixelCount()
        self._view.contentsView.detectorPixelCountCheckBox.setVisible(canSyncDetectorPixelCount)
        self._view.contentsView.detectorPixelCountCheckBox.setChecked(canSyncDetectorPixelCount)

        canSyncDetectorPixelSize = self._presenter.canSyncDetectorPixelSize()
        self._view.contentsView.detectorPixelSizeCheckBox.setVisible(canSyncDetectorPixelSize)
        self._view.contentsView.detectorPixelSizeCheckBox.setChecked(canSyncDetectorPixelSize)

        canSyncDetectorDistance = self._presenter.canSyncDetectorDistance()
        self._view.contentsView.detectorDistanceCheckBox.setVisible(canSyncDetectorDistance)
        self._view.contentsView.detectorDistanceCheckBox.setChecked(canSyncDetectorDistance)

        canSyncPatternCropCenter = self._presenter.canSyncPatternCropCenter()
        self._view.contentsView.patternCropCenterCheckBox.setVisible(canSyncPatternCropCenter)
        self._view.contentsView.patternCropCenterCheckBox.setChecked(canSyncPatternCropCenter)

        canSyncPatternCropExtent = self._presenter.canSyncPatternCropExtent()
        self._view.contentsView.patternCropExtentCheckBox.setVisible(canSyncPatternCropExtent)
        self._view.contentsView.patternCropExtentCheckBox.setChecked(canSyncPatternCropExtent)

        canSyncProbeEnergy = self._presenter.canSyncProbeEnergy()
        self._view.contentsView.probeEnergyCheckBox.setVisible(canSyncProbeEnergy)
        self._view.contentsView.probeEnergyCheckBox.setChecked(canSyncProbeEnergy)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
