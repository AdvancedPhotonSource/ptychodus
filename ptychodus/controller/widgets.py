from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtWidgets import QSpinBox

from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.parametric import IntegerParameter, RealParameter
from ..view.widgets import DecimalSlider, LengthWidget

logger = logging.getLogger(__name__)


class SpinBoxParameterController(Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, parameter: IntegerParameter, spinBox: QSpinBox) -> None:
        self._parameter = parameter
        self._spinBox = spinBox

        spinBox.valueChanged.connect(parameter.setValue)

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None or maximum is None:
            logger.error('Range not provided!')
        else:
            self._spinBox.blockSignals(True)
            self._spinBox.setRange(minimum, maximum)
            self._spinBox.setValue(self._parameter.getValue())
            self._spinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class DecimalSliderParameterController(Observer):

    def __init__(self, parameter: RealParameter, slider: DecimalSlider) -> None:
        self._parameter = parameter
        self._slider = slider

        slider.valueChanged.connect(parameter.setValue)

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None or maximum is None:
            logger.error('Range not provided!')
        else:
            value = Decimal(repr(self._parameter.getValue()))
            range_ = Interval[Decimal](Decimal(repr(minimum)), Decimal(repr(maximum)))
            self._slider.setValueAndRange(value, range_)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class LengthWidgetParameterController(Observer):

    def __init__(self, parameter: RealParameter, widget: LengthWidget) -> None:
        self._parameter = parameter
        self._widget = widget

        widget.lengthChanged.connect(self._syncViewToModel)

    def _syncViewToModel(self, value: Decimal) -> None:
        self._parameter.setValue(float(value))

    def _syncModelToView(self) -> None:
        self._widget.setLengthInMeters(Decimal(repr(self._parameter.getValue())))

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()
