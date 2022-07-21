from __future__ import annotations
from typing import Callable
import colorsys

import numpy

from ...api.image import RealArrayType
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import VisualizationArrayComponent

CylindricalColorModel = Callable[[float, float, float], tuple[float, float, float]]


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, componentChooser: PluginChooser[VisualizationArrayComponent],
                 displayRange: DisplayRange, name: str, model: CylindricalColorModel,
                 variant: bool) -> None:
        super().__init__(name, componentChooser, displayRange)
        self._model = numpy.vectorize(model)
        self._variant = variant

    @classmethod
    def createVariants(cls, componentChooser: PluginChooser[VisualizationArrayComponent],
                       displayRange: DisplayRange) -> list[Colorizer]:
        return [
            cls(componentChooser, displayRange, 'HSV Saturation', colorsys.hsv_to_rgb, False),
            cls(componentChooser, displayRange, 'HSV Value', colorsys.hsv_to_rgb, True),
            cls(componentChooser, displayRange, 'HLS Lightness', colorsys.hls_to_rgb, False),
            cls(componentChooser, displayRange, 'HLS Saturation', colorsys.hls_to_rgb, True)
        ]

    def __call__(self) -> RealArrayType:
        # TODO apply displayRange
        phaseInRadians = numpy.angle(self._component.getArray())
        h = (phaseInRadians + numpy.pi) / (2 * numpy.pi)
        x = self._component()
        y = numpy.zeros_like(h)

        if self._variant:
            y, x = x, y

        return numpy.stack(self._model(h, x, y), axis=-1)
