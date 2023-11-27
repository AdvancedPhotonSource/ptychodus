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

        view.widthInPixelsSpinBox.valueChanged.connect(presenter.setWidthInPixels)
        view.heightInPixelsSpinBox.valueChanged.connect(presenter.setHeightInPixels)
        view.pixelWidthWidget.lengthChanged.connect(presenter.setPixelWidthInMeters)
        view.pixelHeightWidget.lengthChanged.connect(presenter.setPixelHeightInMeters)
        view.bitDepthSpinBox.valueChanged.connect(presenter.setBitDepth)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.widthInPixelsSpinBox.blockSignals(True)
        self._view.widthInPixelsSpinBox.setRange(self._presenter.getWidthInPixelsLimits().lower,
                                                 self._presenter.getWidthInPixelsLimits().upper)
        self._view.widthInPixelsSpinBox.setValue(self._presenter.getWidthInPixels())
        self._view.widthInPixelsSpinBox.blockSignals(False)

        self._view.heightInPixelsSpinBox.blockSignals(True)
        self._view.heightInPixelsSpinBox.setRange(self._presenter.getHeightInPixelsLimits().lower,
                                                  self._presenter.getHeightInPixelsLimits().upper)
        self._view.heightInPixelsSpinBox.setValue(self._presenter.getHeightInPixels())
        self._view.heightInPixelsSpinBox.blockSignals(False)

        self._view.pixelWidthWidget.setLengthInMeters(self._presenter.getPixelWidthInMeters())
        self._view.pixelHeightWidget.setLengthInMeters(self._presenter.getPixelHeightInMeters())

        self._view.bitDepthSpinBox.blockSignals(True)
        self._view.bitDepthSpinBox.setRange(self._presenter.getBitDepthLimits().lower,
                                            self._presenter.getBitDepthLimits().upper)
        self._view.bitDepthSpinBox.setValue(self._presenter.getBitDepth())
        self._view.bitDepthSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
