from abc import ABC, abstractmethod
from collections.abc import Sequence
import colorsys

from matplotlib.colors import Normalize
import numpy

from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import VisualizationArray


class CylindricalColorModel(ABC):

    @abstractmethod
    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        pass

    @abstractmethod
    def getHue(self, h: float) -> tuple[float, float, float, float]:
        pass


class HSVSaturationColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, x, 1.0), 1.0

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 1.0)


class HSVValueColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, 1.0, x), 1.0

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 1.0)


class HSVAlphaColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hsv_to_rgb(h, 1.0, 1.0), x

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 1.0)


class HLSLightnessColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, x, 1.0), 1.0

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 0.5)


class HLSSaturationColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, 0.5, x), 1.0

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 1.0)


class HLSAlphaColorModel(CylindricalColorModel):

    def __call__(self, h: float, x: float) -> tuple[float, float, float, float]:
        return *colorsys.hls_to_rgb(h, 0.5, 1.0), x

    def getHue(self, h: float) -> tuple[float, float, float, float]:
        return self(h, 1.0)


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation],
                 variantChooser: PluginChooser[CylindricalColorModel]) -> None:
        super().__init__(array, displayRange, transformChooser)
        self._variantChooser = variantChooser
        self._variantChooser.addObserver(self)

    @classmethod
    def createColorizerVariants(
            cls, array: VisualizationArray, displayRange: DisplayRange,
            transformChooser: PluginChooser[ScalarTransformation]) -> Sequence[Colorizer]:
        variantChooser = PluginChooser[CylindricalColorModel]()
        variantChooser.registerPlugin(
            HSVSaturationColorModel(),
            simpleName='HSV-S',
            displayName='HSV Saturation',
        )
        variantChooser.registerPlugin(
            HSVValueColorModel(),
            simpleName='HSV-V',
            displayName='HSV Value',
        )
        variantChooser.registerPlugin(
            HSVAlphaColorModel(),
            simpleName='HSV-A',
            displayName='HSV Alpha',
        )
        variantChooser.registerPlugin(
            HLSLightnessColorModel(),
            simpleName='HLS-L',
            displayName='HLS Lightness',
        )
        variantChooser.registerPlugin(
            HLSSaturationColorModel(),
            simpleName='HLS-S',
            displayName='HLS Saturation',
        )
        variantChooser.registerPlugin(
            HLSAlphaColorModel(),
            simpleName='HLS-A',
            displayName='HLS Alpha',
        )
        variantChooser.setCurrentPluginByName('HSV-V')
        return [cls(array, displayRange, transformChooser, variantChooser)]

    @property
    def name(self) -> str:
        return 'Complex'

    def getVariantNameList(self) -> Sequence[str]:
        return self._variantChooser.getDisplayNameList()

    def getVariantName(self) -> str:
        return self._variantChooser.currentPlugin.displayName

    def setVariantByName(self, name: str) -> None:
        self._variantChooser.setCurrentPluginByName(name)

    def getColorSamples(self, normalizedValues: RealArrayType) -> RealArrayType:
        model = numpy.vectorize(self._variantChooser.currentPlugin.strategy.getHue)
        r, g, b, a = model(normalizedValues)
        return numpy.stack((r, g, b, a), axis=-1)

    def isCyclic(self) -> bool:
        return True

    def getDataArray(self) -> RealArrayType:
        transform = self._transformChooser.currentPlugin.strategy
        values = self._array.getAmplitude()
        return transform(values)

    def __call__(self) -> RealArrayType:
        if self._displayRange.getUpper() <= self._displayRange.getLower():
            return numpy.zeros((*self._array.shape, 4))

        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)

        model = numpy.vectorize(self._variantChooser.currentPlugin.strategy)
        h = (self._array.getPhaseInRadians() + numpy.pi) / (2 * numpy.pi)
        x = norm(self.getDataArray())
        r, g, b, a = model(h, x)

        return numpy.stack((r, g, b, a), axis=-1)

    def update(self, observable: Observable) -> None:
        if observable is self._variantChooser:
            self.notifyObservers()
        else:
            super().update(observable)
