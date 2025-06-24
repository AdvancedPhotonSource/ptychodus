from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Final

import matplotlib.colors
import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.typing import RealArrayType

__all__ = [
    'CylindricalColorModel',
    'CylindricalColorModelParameter',
]


def hsva_to_rgba(
    hue: RealArrayType, saturation: RealArrayType, value: RealArrayType, alpha: RealArrayType
) -> RealArrayType:
    hsv = numpy.stack((hue, saturation, value), axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    if alpha.ndim == 1:
        return numpy.column_stack((rgb, alpha))

    return numpy.dstack((rgb, alpha))


def _v(m1: RealArrayType, m2: RealArrayType, hue: RealArrayType) -> RealArrayType:
    """Adapted from colorsys._v in the Python standard library."""
    if m1.shape != hue.shape or m2.shape != hue.shape:
        raise ValueError('Shape mismatch: m1, m2, and hue must have the same shape.')

    hue = hue % 1.0

    return numpy.select(
        [hue < 1.0 / 6.0, hue < 0.5, hue < 2.0 / 3.0],
        [m1 + (m2 - m1) * hue * 6.0, m2, m1 + (m2 - m1) * (2.0 / 3.0 - hue) * 6.0],
        default=m1,
    )


def hlsa_to_rgba(
    hue: RealArrayType, lightness: RealArrayType, saturation: RealArrayType, alpha: RealArrayType
) -> RealArrayType:
    """Adapted from colorsys.hls_to_rgb in the Python standard library."""
    one_third: Final[float] = 1.0 / 3.0

    m2 = numpy.where(
        lightness <= 0.5,
        lightness * (1.0 + saturation),
        lightness + saturation - (lightness * saturation),
    )
    m1 = 2.0 * lightness - m2

    red = numpy.where(saturation > 0.0, _v(m1, m2, hue + one_third), lightness)
    green = numpy.where(saturation > 0.0, _v(m1, m2, hue), lightness)
    blue = numpy.where(saturation > 0.0, _v(m1, m2, hue - one_third), lightness)

    return numpy.stack((red, green, blue, alpha), axis=-1)


class CylindricalColorModel(ABC):
    @abstractmethod
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        pass


class HSVSaturationColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hsva_to_rgba(h, x, ones, ones)


class HSVValueColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hsva_to_rgba(h, ones, x, ones)


class HSVAlphaColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hsva_to_rgba(h, ones, ones, x)


class HLSLightnessColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hlsa_to_rgba(h, x, ones, ones)


class HLSSaturationColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hlsa_to_rgba(h, ones / 2.0, x, ones)


class HLSAlphaColorModel(CylindricalColorModel):
    def __call__(self, h: RealArrayType, x: RealArrayType) -> RealArrayType:
        ones = numpy.ones_like(h)
        return hlsa_to_rgba(h, ones / 2.0, ones, x)


class CylindricalColorModelParameter(Parameter[str], Observer):
    def __init__(self) -> None:
        super().__init__()
        self._chooser = PluginChooser[CylindricalColorModel]()
        self._chooser.register_plugin(
            HSVSaturationColorModel(),
            simple_name='HSV-S',
            display_name='HSV Saturation',
        )
        self._chooser.register_plugin(
            HSVValueColorModel(),
            simple_name='HSV-V',
            display_name='HSV Value',
        )
        self._chooser.register_plugin(
            HSVAlphaColorModel(),
            simple_name='HSV-A',
            display_name='HSV Alpha',
        )
        self._chooser.register_plugin(
            HLSLightnessColorModel(),
            simple_name='HLS-L',
            display_name='HLS Lightness',
        )
        self._chooser.register_plugin(
            HLSSaturationColorModel(),
            simple_name='HLS-S',
            display_name='HLS Saturation',
        )
        self._chooser.register_plugin(
            HLSAlphaColorModel(),
            simple_name='HLS-A',
            display_name='HLS Alpha',
        )
        self.set_value('HSV-V')
        self._chooser.add_observer(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def get_value(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def set_value(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def copy(self) -> Parameter[str]:
        parameter = CylindricalColorModelParameter()
        parameter.set_value(self.get_value())
        return parameter

    def get_plugin(self) -> CylindricalColorModel:
        return self._chooser.get_current_plugin().strategy

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
