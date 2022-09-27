from __future__ import annotations

from ...model import CropPresenter, DiffractionDatasetPresenter, Observable, Observer
from ...view import DataNavigationPage, PatternCropView, PatternsView


class PatternCropController(Observer):

    def __init__(self, presenter: CropPresenter, view: PatternCropView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: CropPresenter,
                       view: PatternCropView) -> PatternCropController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setCropEnabled)

        view.centerXSpinBox.valueChanged.connect(presenter.setCenterXInPixels)
        view.centerYSpinBox.valueChanged.connect(presenter.setCenterYInPixels)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentXInPixels)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentYInPixels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isCropEnabled())

        self._view.centerXSpinBox.blockSignals(True)
        self._view.centerXSpinBox.setRange(self._presenter.getMinCenterXInPixels(),
                                           self._presenter.getMaxCenterXInPixels())
        self._view.centerXSpinBox.setValue(self._presenter.getCenterXInPixels())
        self._view.centerXSpinBox.blockSignals(False)

        self._view.centerYSpinBox.blockSignals(True)
        self._view.centerYSpinBox.setRange(self._presenter.getMinCenterYInPixels(),
                                           self._presenter.getMaxCenterYInPixels())
        self._view.centerYSpinBox.setValue(self._presenter.getCenterYInPixels())
        self._view.centerYSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getMinExtentXInPixels(),
                                           self._presenter.getMaxExtentXInPixels())
        self._view.extentXSpinBox.setValue(self._presenter.getExtentXInPixels())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getMinExtentYInPixels(),
                                           self._presenter.getMaxExtentYInPixels())
        self._view.extentYSpinBox.setValue(self._presenter.getExtentYInPixels())
        self._view.extentYSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternsController:

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter, cropPresenter: CropPresenter,
                 view: DataNavigationPage[PatternsView]) -> None:
        self._datasetPresenter = datasetPresenter
        self._cropPresenter = cropPresenter
        self._view = view
        self._cropController = PatternCropController.createInstance(cropPresenter,
                                                                    view.contentsView.cropView)

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       cropPresenter: CropPresenter,
                       view: DataNavigationPage[PatternsView]) -> PatternsController:
        controller = cls(datasetPresenter, cropPresenter, view)
        view.forwardButton.clicked.connect(datasetPresenter.processDiffractionPatterns)
        return controller
