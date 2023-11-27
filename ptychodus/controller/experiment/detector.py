from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.experiment import DetectorPresenter
from ...view.experiment import DetectorView


class DetectorController(Observer):

    def __init__(self, presenter: DetectorPresenter, view: DetectorView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorPresenter,
                       view: DetectorView) -> DetectorController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfPixelsXSpinBox.valueChanged.connect(presenter.setNumberOfPixelsX)
        view.numberOfPixelsYSpinBox.valueChanged.connect(presenter.setNumberOfPixelsY)
        view.pixelSizeXWidget.lengthChanged.connect(presenter.setPixelSizeXInMeters)
        view.pixelSizeYWidget.lengthChanged.connect(presenter.setPixelSizeYInMeters)
        view.bitDepthSpinBox.valueChanged.connect(presenter.setBitDepth)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPixelsXSpinBox.blockSignals(True)
        self._view.numberOfPixelsXSpinBox.setRange(
            self._presenter.getNumberOfPixelsXLimits().lower,
            self._presenter.getNumberOfPixelsXLimits().upper)
        self._view.numberOfPixelsXSpinBox.setValue(self._presenter.getNumberOfPixelsX())
        self._view.numberOfPixelsXSpinBox.blockSignals(False)

        self._view.numberOfPixelsYSpinBox.blockSignals(True)
        self._view.numberOfPixelsYSpinBox.setRange(
            self._presenter.getNumberOfPixelsYLimits().lower,
            self._presenter.getNumberOfPixelsYLimits().upper)
        self._view.numberOfPixelsYSpinBox.setValue(self._presenter.getNumberOfPixelsY())
        self._view.numberOfPixelsYSpinBox.blockSignals(False)

        self._view.pixelSizeXWidget.setLengthInMeters(self._presenter.getPixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(self._presenter.getPixelSizeYInMeters())

        self._view.bitDepthSpinBox.blockSignals(True)
        self._view.bitDepthSpinBox.setRange(self._presenter.getBitDepthLimits().lower,
                                            self._presenter.getBitDepthLimits().upper)
        self._view.bitDepthSpinBox.setValue(self._presenter.getBitDepth())
        self._view.bitDepthSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
