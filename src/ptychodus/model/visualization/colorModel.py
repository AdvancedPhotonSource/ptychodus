from abc import ABC, abstractmethod
from collections.abc import Iterator
import colorsys

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser

__all__ = [
    'CylindricalColorModel',
    'CylindricalColorModelParameter',
]


class CylindricalColorModel(ABC):
    @abstractmethod
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        pass


class HSVSaturationColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, x, 1.0), 1.0


class HSVValueColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, 1.0, x), 1.0


class HSVAlphaColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, 1.0, 1.0), x


class HLSLightnessColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, x, 1.0), 1.0


class HLSSaturationColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, 0.5, x), 1.0


class HLSAlphaColorModel(CylindricalColorModel):
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, 0.5, 1.0), x


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
        self.setValue('HSV-V')
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def getValue(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def getValueAsString(self) -> str:
        return self.getValue()

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def copy(self) -> Parameter[str]:
        parameter = CylindricalColorModelParameter()
        parameter.setValue(self.getValue())
        return parameter

    def getPlugin(self) -> CylindricalColorModel:
        return self._chooser.get_current_plugin().strategy

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
