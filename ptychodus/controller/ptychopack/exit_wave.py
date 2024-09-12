from __future__ import annotations
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.ptychopack import PtychoPackPresenter
from ...view.ptychopack import PtychoPackExitWaveCorrectionView

logger = logging.getLogger(__name__)


class PtychoPackExitWaveCorrectionController(Observer):

    def __init__(self, presenter: PtychoPackPresenter,
                 view: PtychoPackExitWaveCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        view.dm_relaxation_slider.valueChanged.connect(presenter.set_dm_exit_wave_relaxation)
        view.raar_relaxation_slider.valueChanged.connect(presenter.set_raar_exit_wave_relaxation)

        self._sync_model_to_view()
        presenter.addObserver(self)

    def _sync_model_to_view(self) -> None:
        self._view.dm_relaxation_slider.setValueAndRange(
            self._presenter.get_dm_exit_wave_relaxation(),
            self._presenter.get_dm_exit_wave_relaxation_limits(),
            blockValueChangedSignal=True)
        self._view.raar_relaxation_slider.setValueAndRange(
            self._presenter.get_raar_exit_wave_relaxation(),
            self._presenter.get_raar_exit_wave_relaxation_limits(),
            blockValueChangedSignal=True)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
