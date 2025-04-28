from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter, StringParameter

from ...model.ptychi import PtyChiEnumerators
from ..parametric import (
    ComboBoxParameterViewController,
    SpinBoxParameterViewController,
    ParameterViewController,
)

__all__ = [
    'PtyChiOptimizationPlanViewController',
    'PtyChiOptimizerParameterViewController',
]


class PtyChiStopSpinBoxParameterViewController(ParameterViewController, Observer):
    def __init__(
        self, stop: IntegerParameter, num_epochs: IntegerParameter, *, tool_tip: str = ''
    ) -> None:
        super().__init__()
        self._stop = stop
        self._num_epochs = num_epochs
        self._widget = QSpinBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._sync_model_to_view()
        self._widget.valueChanged.connect(self._sync_view_to_model)
        stop.add_observer(self)
        num_epochs.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, value: int) -> None:
        num_epochs = self._num_epochs.get_value()
        self._stop.set_value(value if value < num_epochs else -1)

    def _sync_model_to_view(self) -> None:
        num_epochs = self._num_epochs.get_value()
        stop = self._stop.get_value()

        self._widget.blockSignals(True)
        self._widget.setRange(0, num_epochs)
        self._widget.setValue(num_epochs if stop < 0 else stop)
        self._widget.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable in (self._stop, self._num_epochs):
            self._sync_model_to_view()


class PtyChiOptimizationPlanViewController(ParameterViewController):
    def __init__(
        self,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__()
        self._start_view_controller = SpinBoxParameterViewController(
            start, tool_tip='Iteration to start optimizing'
        )
        self._stop_view_controller = PtyChiStopSpinBoxParameterViewController(
            stop, num_epochs, tool_tip='Iteration to stop optimizing'
        )
        self._stride_view_controller = SpinBoxParameterViewController(
            stride, tool_tip='Number of iterations between updates'
        )
        self._widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('Start'), 0)
        layout.addWidget(self._start_view_controller.get_widget(), 1)
        layout.addWidget(QLabel('Stop'), 0)
        layout.addWidget(self._stop_view_controller.get_widget(), 1)
        layout.addWidget(QLabel('Stride'), 0)
        layout.addWidget(self._stride_view_controller.get_widget(), 1)
        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget


class PtyChiOptimizerParameterViewController(ComboBoxParameterViewController):
    def __init__(self, parameter: StringParameter, enumerators: PtyChiEnumerators) -> None:
        super().__init__(parameter, enumerators.optimizers(), tool_tip='Name of the optimizer.')
