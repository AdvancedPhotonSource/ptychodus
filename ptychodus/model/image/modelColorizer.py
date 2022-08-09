from __future__ import annotations
from typing import Callable
import colorsys

from matplotlib.colors import Normalize
import numpy

from ...api.image import RealArrayType
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import VisualizationArrayComponent

CylindricalColorModel = Callable[[float, float, float], tuple[float, float, float]]


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, name: str, componentChooser: PluginChooser[VisualizationArrayComponent],
                 displayRange: DisplayRange, model: CylindricalColorModel, variant: bool) -> None:
        super().__init__(name, componentChooser, displayRange)
        self._model = numpy.vectorize(model)
        self._variant = variant

    @classmethod
    def createVariants(cls, componentChooser: PluginChooser[VisualizationArrayComponent],
                       displayRange: DisplayRange) -> list[Colorizer]:
        return [
            cls('HSV Saturation', componentChooser, displayRange, colorsys.hsv_to_rgb, False),
            cls('HSV Value', componentChooser, displayRange, colorsys.hsv_to_rgb, True),
            cls('HLS Lightness', componentChooser, displayRange, colorsys.hls_to_rgb, False),
            cls('HLS Saturation', componentChooser, displayRange, colorsys.hls_to_rgb, True)
        ]

    def getVariantList(self) -> list[str]:
        return list() # FIXME

    def getVariant(self) -> str:
        return str() # FIXME

    def setVariant(self, name: str) -> None:
        pass # FIXME

    def getDataRange(self) -> Interval[Decimal]:
        pass # FIXME

    def __call__(self) -> RealArrayType:
        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)

        phaseInRadians = numpy.angle(self._component.getArray())
        h = (phaseInRadians + numpy.pi) / (2 * numpy.pi)
        x = norm(self._component())
        y = numpy.ones_like(h)
        a = numpy.ones_like(h)

        if self._variant:
            y, x = x, y

        r, g, b = self._model(h, x, y)

        return numpy.stack((r, g, b, a), axis=-1)
