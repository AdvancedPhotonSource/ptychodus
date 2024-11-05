from PyQt5.QtWidgets import QHBoxLayout, QWidget

from ptychodus.api.parametric import IntegerParameter, StringParameter

from ...model.pty_chi import PtyChiEnumerators
from ..parametric import (
    ComboBoxParameterViewController,
    IntegerLineEditParameterViewController,
    ParameterViewController,
)


class PtyChiOptimizationPlanViewController(ParameterViewController):
    def __init__(
        self, start: IntegerParameter, stop: IntegerParameter, stride: IntegerParameter
    ) -> None:
        super().__init__()
        self._startViewController = IntegerLineEditParameterViewController(
            start, tool_tip='Iteration to start optimizing'
        )
        self._stopViewController = IntegerLineEditParameterViewController(
            stop, tool_tip='Iteration to stop optimizing'
        )
        self._strideViewController = IntegerLineEditParameterViewController(
            stride, tool_tip='Number of iterations between updates'
        )
        self._widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._startViewController.getWidget())
        layout.addWidget(self._stopViewController.getWidget())
        layout.addWidget(self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtyChiOptimizerParameterViewController(ComboBoxParameterViewController):
    def __init__(self, parameter: StringParameter, enumerators: PtyChiEnumerators) -> None:
        super().__init__(parameter, enumerators.optimizers())
