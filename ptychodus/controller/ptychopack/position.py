from __future__ import annotations
import logging

from ptychodus.api.observer import Observable, Observer

from ...model.ptychopack import PtychoPackPresenter
from ...view.ptychopack import PtychoPackPositionCorrectionView

logger = logging.getLogger(__name__)


class PtychoPackPositionCorrectionController(Observer):

    def __init__(self, presenter: PtychoPackPresenter,
                 view: PtychoPackPositionCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        view.plan_widget.start_lineedit.editingFinished.connect(self._sync_plan_start_to_model)
        view.plan_widget.stop_lineedit.editingFinished.connect(self._sync_plan_stop_to_model)
        view.plan_widget.stride_lineedit.editingFinished.connect(self._sync_plan_stride_to_model)
        view.probe_threshold_slider.valueChanged.connect(
            presenter.set_position_correction_probe_threshold)
        view.feedback_line_edit.valueChanged.connect(presenter.set_position_correction_feedback)

        self._sync_model_to_view()
        presenter.addObserver(self)

    def _sync_plan_start_to_model(self) -> None:
        text = self._view.plan_widget.start_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_position_correction_plan_start(value)

    def _sync_plan_stop_to_model(self) -> None:
        text = self._view.plan_widget.stop_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_position_correction_plan_stop(value)

    def _sync_plan_stride_to_model(self) -> None:
        text = self._view.plan_widget.stride_lineedit.text()

        try:
            value = int(text)
        except ValueError:
            logger.warning(f'Failed to convert \"{text}\" to int!')
        else:
            self._presenter.set_position_correction_plan_stride(value)

    def _sync_model_to_view(self) -> None:
        self._view.plan_widget.start_lineedit.setText(
            str(self._presenter.get_position_correction_plan_start()))
        self._view.plan_widget.stop_lineedit.setText(
            str(self._presenter.get_position_correction_plan_stop()))
        self._view.plan_widget.stride_lineedit.setText(
            str(self._presenter.get_position_correction_plan_stride()))

        self._view.probe_threshold_slider.setValueAndRange(
            self._presenter.get_position_correction_probe_threshold(),
            self._presenter.get_position_correction_probe_threshold_limits(),
            blockValueChangedSignal=True)
        self._view.feedback_line_edit.setValue(self._presenter.get_position_correction_feedback())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()
