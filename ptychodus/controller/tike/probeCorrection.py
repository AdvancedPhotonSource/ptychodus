from __future__ import annotations

from ptychodus.api.observer import Observable, Observer

from ...model.tike import TikeProbeCorrectionPresenter
from ...view.tike import TikeProbeCorrectionView
from .adaptiveMoment import TikeAdaptiveMomentController
from .probeSupport import TikeProbeSupportController


class TikeProbeCorrectionController(Observer):
    def __init__(
        self, presenter: TikeProbeCorrectionPresenter, view: TikeProbeCorrectionView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._probeSupportController = TikeProbeSupportController.createInstance(
            presenter, view.probeSupportView
        )
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView
        )

    @classmethod
    def createInstance(
        cls, presenter: TikeProbeCorrectionPresenter, view: TikeProbeCorrectionView
    ) -> TikeProbeCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setProbeCorrectionEnabled)

        view.forceSparsitySlider.valueChanged.connect(presenter.setForceSparsity)
        view.forceOrthogonalityCheckBox.toggled.connect(presenter.setForceOrthogonalityEnabled)
        view.forceCenteredIntensityCheckBox.toggled.connect(
            presenter.setForceCenteredIntensityEnabled
        )
        view.additionalProbePenaltyLineEdit.valueChanged.connect(
            presenter.setAdditionalProbePenalty
        )

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isProbeCorrectionEnabled())

        self._view.forceSparsitySlider.setValueAndRange(
            self._presenter.getForceSparsity(),
            self._presenter.getForceSparsityLimits(),
            blockValueChangedSignal=True,
        )
        self._view.forceOrthogonalityCheckBox.setChecked(
            self._presenter.isForceOrthogonalityEnabled()
        )
        self._view.forceCenteredIntensityCheckBox.setChecked(
            self._presenter.isForceCenteredIntensityEnabled()
        )
        self._view.additionalProbePenaltyLineEdit.setMinimum(
            self._presenter.getAdditionalProbePenaltyMinimum()
        )
        self._view.additionalProbePenaltyLineEdit.setValue(
            self._presenter.getAdditionalProbePenalty()
        )

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
