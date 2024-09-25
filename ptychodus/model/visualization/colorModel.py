from abc import ABC, abstractmethod
from collections.abc import Iterator
import colorsys

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter, ParameterGroup
from ptychodus.api.plugins import PluginChooser

__all__ = [
    "CylindricalColorModel",
    "CylindricalColorModelParameter",
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

    def __init__(self, parent: ParameterGroup) -> None:
        super().__init__()
        self._chooser = PluginChooser[CylindricalColorModel]()
        self._chooser.registerPlugin(
            HSVSaturationColorModel(),
            simpleName="HSV-S",
            displayName="HSV Saturation",
        )
        self._chooser.registerPlugin(
            HSVValueColorModel(),
            simpleName="HSV-V",
            displayName="HSV Value",
        )
        self._chooser.registerPlugin(
            HSVAlphaColorModel(),
            simpleName="HSV-A",
            displayName="HSV Alpha",
        )
        self._chooser.registerPlugin(
            HLSLightnessColorModel(),
            simpleName="HLS-L",
            displayName="HLS Lightness",
        )
        self._chooser.registerPlugin(
            HLSSaturationColorModel(),
            simpleName="HLS-S",
            displayName="HLS Saturation",
        )
        self._chooser.registerPlugin(
            HLSAlphaColorModel(),
            simpleName="HLS-A",
            displayName="HLS Alpha",
        )
        self.setValue("HSV-V")
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for name in self._chooser.getDisplayNameList():
            yield name

    def getValue(self) -> str:
        return self._chooser.currentPlugin.displayName

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def getPlugin(self) -> CylindricalColorModel:
        return self._chooser.currentPlugin.strategy

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
