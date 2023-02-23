from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...model.data import (DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter,
                           DiffractionPatternPresenter)
from ...view import (DataNavigationPage, PatternCropView, PatternLoadView, PatternTransformView,
                     PatternsView)
from ..data import FileDialogFactory


class PatternLoadController(Observer):

    def __init__(self, presenter: DiffractionDatasetPresenter, view: PatternLoadView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DiffractionDatasetPresenter,
                       view: PatternLoadView) -> PatternLoadController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfThreadsSpinBox.valueChanged.connect(presenter.setNumberOfDataThreads)
        view.memmapCheckBox.toggled.connect(presenter.setMemmapEnabled)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfThreadsSpinBox.blockSignals(True)
        self._view.numberOfThreadsSpinBox.setRange(
            self._presenter.getNumberOfDataThreadsLimits().lower,
            self._presenter.getNumberOfDataThreadsLimits().upper)
        self._view.numberOfThreadsSpinBox.setValue(self._presenter.getNumberOfDataThreads())
        self._view.numberOfThreadsSpinBox.blockSignals(False)

        self._view.memmapCheckBox.setChecked(self._presenter.isMemmapEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternCropController(Observer):

    def __init__(self, presenter: DiffractionPatternPresenter, view: PatternCropView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DiffractionPatternPresenter,
                       view: PatternCropView) -> PatternCropController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setCropEnabled)

        view.centerXSpinBox.valueChanged.connect(presenter.setCropCenterXInPixels)
        view.centerYSpinBox.valueChanged.connect(presenter.setCropCenterYInPixels)
        view.extentXSpinBox.valueChanged.connect(presenter.setCropExtentXInPixels)
        view.extentYSpinBox.valueChanged.connect(presenter.setCropExtentYInPixels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isCropEnabled())

        self._view.centerXSpinBox.blockSignals(True)
        self._view.centerXSpinBox.setRange(self._presenter.getCropCenterXLimitsInPixels().lower,
                                           self._presenter.getCropCenterXLimitsInPixels().upper)
        self._view.centerXSpinBox.setValue(self._presenter.getCropCenterXInPixels())
        self._view.centerXSpinBox.blockSignals(False)

        self._view.centerYSpinBox.blockSignals(True)
        self._view.centerYSpinBox.setRange(self._presenter.getCropCenterYLimitsInPixels().lower,
                                           self._presenter.getCropCenterYLimitsInPixels().upper)
        self._view.centerYSpinBox.setValue(self._presenter.getCropCenterYInPixels())
        self._view.centerYSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getCropExtentXLimitsInPixels().lower,
                                           self._presenter.getCropExtentXLimitsInPixels().upper)
        self._view.extentXSpinBox.setValue(self._presenter.getCropExtentXInPixels())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getCropExtentYLimitsInPixels().lower,
                                           self._presenter.getCropExtentYLimitsInPixels().upper)
        self._view.extentYSpinBox.setValue(self._presenter.getCropExtentYInPixels())
        self._view.extentYSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternTransformController(Observer):

    def __init__(self, presenter: DiffractionPatternPresenter, view: PatternTransformView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DiffractionPatternPresenter,
                       view: PatternTransformView) -> PatternTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.thresholdCheckBox.toggled.connect(presenter.setThresholdEnabled)
        view.thresholdSpinBox.valueChanged.connect(presenter.setThresholdValue)
        view.flipXCheckBox.toggled.connect(presenter.setFlipXEnabled)
        view.flipYCheckBox.toggled.connect(presenter.setFlipYEnabled)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.thresholdCheckBox.setChecked(self._presenter.isThresholdEnabled())

        self._view.thresholdSpinBox.blockSignals(True)
        self._view.thresholdSpinBox.setRange(self._presenter.getThresholdValueLimits().lower,
                                             self._presenter.getThresholdValueLimits().upper)
        self._view.thresholdSpinBox.setValue(self._presenter.getThresholdValue())
        self._view.thresholdSpinBox.blockSignals(False)

        self._view.flipXCheckBox.setChecked(self._presenter.isFlipXEnabled())
        self._view.flipYCheckBox.setChecked(self._presenter.isFlipYEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternsController:

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 patternPresenter: DiffractionPatternPresenter,
                 view: DataNavigationPage[PatternsView],
                 fileDialogFactory: FileDialogFactory) -> None:
        self._datasetPresenter = datasetPresenter
        self._patternPresenter = patternPresenter
        self._view = view
        self._loadController = PatternLoadController.createInstance(datasetPresenter,
                                                                    view.contentsView.loadView)
        self._cropController = PatternCropController.createInstance(patternPresenter,
                                                                    view.contentsView.cropView)
        self._transformController = PatternTransformController.createInstance(
            patternPresenter, view.contentsView.transformView)

    @classmethod
    def createInstance(cls, datasetInputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       patternPresenter: DiffractionPatternPresenter,
                       view: DataNavigationPage[PatternsView],
                       fileDialogFactory: FileDialogFactory) -> PatternsController:
        controller = cls(datasetPresenter, patternPresenter, view, fileDialogFactory)
        view.forwardButton.clicked.connect(
            datasetInputOutputPresenter.startProcessingDiffractionPatterns)
        return controller
