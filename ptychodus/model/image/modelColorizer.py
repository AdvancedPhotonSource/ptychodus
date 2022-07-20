from __future__ import annotations
from typing import Callable
import colorsys

import numpy

from ...api.image import RealArrayType
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .visarray import VisualizationArrayComponent

CylindricalColorModel = Callable[[float, float, float], tuple[float, float, float]]


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, componentChooser: PluginChooser[VisualizationArrayComponent], name: str,
                 model: CylindricalColorModel, variant: bool) -> None:
        super().__init__(name, componentChooser)
        self._model = numpy.vectorize(model)
        self._variant = variant
        # FIXME self._displayRange

    @classmethod
    def createVariants(
        cls, componentChooser: PluginChooser[VisualizationArrayComponent]
    ) -> list[CylindricalColorModelColorizer]:
        # FIXME transformChooser is Amplitude only
        return [
            cls(componentChooser, 'HSV Saturation', colorsys.hsv_to_rgb, False),
            cls(componentChooser, 'HSV Value', colorsys.hsv_to_rgb, True),
            cls(componentChooser, 'HLS Lightness', colorsys.hls_to_rgb, False),
            cls(componentChooser, 'HLS Saturation', colorsys.hls_to_rgb, True)
        ]

    def __call__(self) -> RealArrayType:
        component = self._componentChooser.getCurrentStrategy()

        # FIME getPhaseInRadians
        h = (self.getPhaseInRadians() + numpy.pi) / (2 * numpy.pi)
        x = component()
        y = numpy.zeros_like(h)

        if self._variant:
            y, x = x, y

        return numpy.stack(self._model(h, x, y), axis=-1)
