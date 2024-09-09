from __future__ import annotations
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.ptychopack import PtychoPackPresenter
from ...view.ptychopack import PtychoPackObjectCorrectionView

logger = logging.getLogger(__name__)


class PtychoPackObjectCorrectionController(Observer):

    def __init__(self, presenter: PtychoPackPresenter,
                 view: PtychoPackObjectCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        view.plan_widget.start_lineedit.editingFinished.connect(self._sync_plan_start_to_model)
        view.plan_widget.stop_lineedit.editingFinished.connect(self._sync_plan_stop_to_model)
        view.plan_widget.stride_lineedit.editingFinished.connect(self._sync_plan_stride_to_model)
        view.alpha_slider.valueChanged.connect(presenter.set_alpha)
        view.relaxation_slider.valueChanged.connect(presenter.set_object_relaxation)

        self._sync_model_to_view()
        presenter.addObserver(self)

    def _sync_plan_start_to_model(self) -> None:
        text = self._view.plan_widget.start_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_object_correction_plan_start(value)

    def _sync_plan_stop_to_model(self) -> None:
        text = self._view.plan_widget.stop_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_object_correction_plan_stop(value)

    def _sync_plan_stride_to_model(self) -> None:
        text = self._view.plan_widget.stride_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_object_correction_plan_stride(value)

    def _sync_model_to_view(self) -> None:
        self._view.plan_widget.start_lineedit.setText(
            str(self._presenter.get_object_correction_plan_start()))
        self._view.plan_widget.stop_lineedit.setText(
            str(self._presenter.get_object_correction_plan_stop()))
        self._view.plan_widget.stride_lineedit.setText(
            str(self._presenter.get_object_correction_plan_stride()))

        self._view.alpha_slider.setValueAndRange(self._presenter.get_alpha(),
                                                 self._presenter.get_alpha_limits(),
                                                 blockValueChangedSignal=True)
        self._view.relaxation_slider.setValueAndRange(
            self._presenter.get_object_relaxation(),
            self._presenter.get_object_relaxation_limits(),
            blockValueChangedSignal=True)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
