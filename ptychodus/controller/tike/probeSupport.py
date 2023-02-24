from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.tike import TikeProbeCorrectionPresenter
from ...view import TikeProbeSupportView


class TikeProbeSupportController(Observer):

    def __init__(self, presenter: TikeProbeCorrectionPresenter,
                 view: TikeProbeSupportView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: TikeProbeCorrectionPresenter,
                       view: TikeProbeSupportView) -> TikeProbeSupportController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setFiniteProbeSupportEnabled)

        view.weightLineEdit.valueChanged.connect(presenter.setProbeSupportWeight)
        view.radiusSlider.valueChanged.connect(presenter.setProbeSupportRadius)
        view.degreeLineEdit.valueChanged.connect(presenter.setProbeSupportDegree)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isFiniteProbeSupportEnabled())

        self._view.weightLineEdit.setMinimum(self._presenter.getProbeSupportWeightMinimum())
        self._view.weightLineEdit.setValue(self._presenter.getProbeSupportWeight())

        self._view.radiusSlider.setValueAndRange(self._presenter.getProbeSupportRadius(),
                                                 self._presenter.getProbeSupportRadiusLimits(),
                                                 blockValueChangedSignal=True)

        self._view.degreeLineEdit.setMinimum(self._presenter.getProbeSupportDegreeMinimum())
        self._view.degreeLineEdit.setValue(self._presenter.getProbeSupportDegree())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
