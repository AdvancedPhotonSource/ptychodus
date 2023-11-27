from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.experiment import DetectorPresenter
from ...model.probe import ApparatusPresenter
from ...view.detector import DetectorParametersView


class DetectorParametersController(Observer):

    def __init__(self, presenter: DetectorPresenter, apparatusPresenter: ApparatusPresenter,
                 view: DetectorParametersView) -> None:
        super().__init__()
        self._presenter = presenter
        self._apparatusPresenter = apparatusPresenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorPresenter, apparatusPresenter: ApparatusPresenter,
                       view: DetectorParametersView) -> DetectorParametersController:
        controller = cls(presenter, apparatusPresenter, view)
        presenter.addObserver(controller)
        apparatusPresenter.addObserver(controller)

        view.detectorDistanceWidget.lengthChanged.connect(presenter.setDetectorDistanceInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.detectorDistanceWidget.setLengthInMeters(
            self._presenter.getDetectorDistanceInMeters())

        self._view.fresnelNumberWidget.setReadOnly(True)
        self._view.fresnelNumberWidget.setValue(self._apparatusPresenter.getFresnelNumber())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
        elif observable is self._apparatusPresenter:
            self._syncModelToView()
