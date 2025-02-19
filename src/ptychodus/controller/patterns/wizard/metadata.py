from PyQt5.QtWidgets import QWizardPage

from ptychodus.api.observer import Observable, Observer

from ....model.metadata import MetadataPresenter
from ....view.patterns import OpenDatasetWizardMetadataPage


class OpenDatasetWizardMetadataViewController(Observer):
    def __init__(self, presenter: MetadataPresenter) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = OpenDatasetWizardMetadataPage()

        presenter.addObserver(self)
        self._sync_model_to_view()
        self._page._setComplete(True)

    def importMetadata(self) -> None:
        if self._page.detectorPixelCountCheckBox.isChecked():
            self._presenter.syncDetectorPixelCount()

        if self._page.detectorPixelSizeCheckBox.isChecked():
            self._presenter.syncDetectorPixelSize()

        if self._page.detectorBitDepthCheckBox.isChecked():
            self._presenter.syncDetectorBitDepth()

        if self._page.detectorDistanceCheckBox.isChecked():
            self._presenter.syncDetectorDistance()

        self._presenter.syncPatternCrop(
            syncCenter=self._page.patternCropCenterCheckBox.isChecked(),
            syncExtent=self._page.patternCropExtentCheckBox.isChecked(),
        )

        if self._page.probeEnergyCheckBox.isChecked():
            self._presenter.syncProbeEnergy()

    def _sync_model_to_view(self) -> None:
        canSyncDetectorPixelCount = self._presenter.canSyncDetectorPixelCount()
        self._page.detectorPixelCountCheckBox.setVisible(canSyncDetectorPixelCount)
        self._page.detectorPixelCountCheckBox.setChecked(canSyncDetectorPixelCount)

        canSyncDetectorPixelSize = self._presenter.canSyncDetectorPixelSize()
        self._page.detectorPixelSizeCheckBox.setVisible(canSyncDetectorPixelSize)
        self._page.detectorPixelSizeCheckBox.setChecked(canSyncDetectorPixelSize)

        canSyncDetectorBitDepth = self._presenter.canSyncDetectorBitDepth()
        self._page.detectorBitDepthCheckBox.setVisible(canSyncDetectorBitDepth)
        self._page.detectorBitDepthCheckBox.setChecked(canSyncDetectorBitDepth)

        canSyncDetectorDistance = self._presenter.canSyncDetectorDistance()
        self._page.detectorDistanceCheckBox.setVisible(canSyncDetectorDistance)
        self._page.detectorDistanceCheckBox.setChecked(canSyncDetectorDistance)

        canSyncPatternCropCenter = self._presenter.canSyncPatternCropCenter()
        self._page.patternCropCenterCheckBox.setVisible(canSyncPatternCropCenter)
        self._page.patternCropCenterCheckBox.setChecked(canSyncPatternCropCenter)

        canSyncPatternCropExtent = self._presenter.canSyncPatternCropExtent()
        self._page.patternCropExtentCheckBox.setVisible(canSyncPatternCropExtent)
        self._page.patternCropExtentCheckBox.setChecked(canSyncPatternCropExtent)

        canSyncProbeEnergy = self._presenter.canSyncProbeEnergy()
        self._page.probeEnergyCheckBox.setVisible(canSyncProbeEnergy)
        self._page.probeEnergyCheckBox.setChecked(canSyncProbeEnergy)

    def getWidget(self) -> QWizardPage:
        return self._page

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
