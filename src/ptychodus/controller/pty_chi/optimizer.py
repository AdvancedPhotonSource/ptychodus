from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    IntegerParameter,
    StringParameter,
)

from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QWidget,
)

from ..parametric import (
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


class PtyChiOptimizerParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: StringParameter) -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QComboBox()

        self._syncModelToView()
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()
