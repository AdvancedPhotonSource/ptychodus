import logging

import numpy

from ptychodus.api.geometry import Interval
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.visualization import RealArrayType

logger = logging.getLogger(__name__)


class ColorAxis(ParameterGroup):
    def __init__(self) -> None:
        super().__init__()
        self.lower = self.createRealParameter('lower', 0.0)
        self.upper = self.createRealParameter('upper', 1.0)

    def getRange(self) -> Interval[float]:
        return Interval[float].createProper(
            self.lower.getValue(),
            self.upper.getValue(),
        )

    def setRange(self, lower: float, upper: float):
        self.lower.setValue(lower, notify=False)
        self.upper.setValue(upper, notify=False)
        self.notifyObservers()

    def setToDataRange(self, array: RealArrayType) -> None:
        if array.size > 0:
            lower = array.min().item()
            upper = array.max().item()

            if numpy.isfinite(lower) and numpy.isfinite(upper):
                if lower == upper:
                    logger.debug('Array values are uniform.')
                    lower -= 0.5
                    upper += 0.5
            else:
                logger.warning('Array values not finite!')
                lower = 0.0
                upper = 1.0

            self.setRange(lower, upper)
        else:
            logger.warning('Array has zero size!')
