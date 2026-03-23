import logging

from ptychodus.api.parametric import ParameterGroup

logger = logging.getLogger(__name__)


class ColorAxis(ParameterGroup):
    def __init__(self) -> None:
        super().__init__()
        self.lower = self.create_real_parameter('lower', 0.0)
        self.upper = self.create_real_parameter('upper', 1.0)

    def set_range(self, lower: float, upper: float):
        self.lower.set_value(lower, notify=False)
        self.upper.set_value(upper, notify=False)
        self.notify_observers()

    def set_to_data_range(self, lower: float, upper: float) -> None:
        if lower == upper:
            logger.debug('Array values are uniform.')
            lower -= 0.5
            upper += 0.5

        self.set_range(lower, upper)
