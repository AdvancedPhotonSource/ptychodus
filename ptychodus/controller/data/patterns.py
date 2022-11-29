from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import DiffractionDatasetPresenter, DiffractionPatternPresenter
from ...view import (DataNavigationPage, PatternCropView, PatternLoadView, PatternTransformView,
                     PatternsView)


class PatternLoadController(Observer):

    def __init__(self, presenter: DiffractionPatternPresenter, view: PatternLoadView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DiffractionPatternPresenter,
                       view: PatternLoadView) -> PatternLoadController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfThreadsSpinBox.valueChanged.connect(presenter.setNumberOfDataThreads)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfThreadsSpinBox.blockSignals(True)
        self._view.numberOfThreadsSpinBox.setRange(self._presenter.getMinNumberOfDataThreads(),
                                                   self._presenter.getMaxNumberOfDataThreads())
        self._view.numberOfThreadsSpinBox.setValue(self._presenter.getNumberOfDataThreads())
        self._view.numberOfThreadsSpinBox.blockSignals(False)

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
        self._view.centerXSpinBox.setRange(self._presenter.getMinCropCenterXInPixels(),
                                           self._presenter.getMaxCropCenterXInPixels())
        self._view.centerXSpinBox.setValue(self._presenter.getCropCenterXInPixels())
        self._view.centerXSpinBox.blockSignals(False)

        self._view.centerYSpinBox.blockSignals(True)
        self._view.centerYSpinBox.setRange(self._presenter.getMinCropCenterYInPixels(),
                                           self._presenter.getMaxCropCenterYInPixels())
        self._view.centerYSpinBox.setValue(self._presenter.getCropCenterYInPixels())
        self._view.centerYSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getMinCropExtentXInPixels(),
                                           self._presenter.getMaxCropExtentXInPixels())
        self._view.extentXSpinBox.setValue(self._presenter.getCropExtentXInPixels())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getMinCropExtentYInPixels(),
                                           self._presenter.getMaxCropExtentYInPixels())
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

        view.thresholdSpinBox.valueChanged.connect(presenter.setThreshold)
        view.flipXCheckBox.toggled.connect(presenter.setFlipXEnabled)
        view.flipYCheckBox.toggled.connect(presenter.setFlipYEnabled)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.thresholdSpinBox.blockSignals(True)
        self._view.thresholdSpinBox.setRange(self._presenter.getMinThreshold(),
                                             self._presenter.getMaxThreshold())
        self._view.thresholdSpinBox.setValue(self._presenter.getThreshold())
        self._view.thresholdSpinBox.blockSignals(False)

        self._view.flipXCheckBox.setChecked(self._presenter.isFlipXEnabled())
        self._view.flipYCheckBox.setChecked(self._presenter.isFlipYEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternsController:

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 patternPresenter: DiffractionPatternPresenter,
                 view: DataNavigationPage[PatternsView]) -> None:
        self._datasetPresenter = datasetPresenter
        self._patternPresenter = patternPresenter
        self._view = view
        self._loadController = PatternLoadController.createInstance(patternPresenter,
                                                                    view.contentsView.loadView)
        self._cropController = PatternCropController.createInstance(patternPresenter,
                                                                    view.contentsView.cropView)
        self._transformController = PatternTransformController.createInstance(
            patternPresenter, view.contentsView.transformView)

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       patternPresenter: DiffractionPatternPresenter,
                       view: DataNavigationPage[PatternsView]) -> PatternsController:
        controller = cls(datasetPresenter, patternPresenter, view)
        view.forwardButton.clicked.connect(datasetPresenter.startProcessingDiffractionPatterns)
        return controller
