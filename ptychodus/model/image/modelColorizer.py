from abc import ABC, abstractmethod, abstractproperty
from decimal import Decimal
import colorsys

from matplotlib.colors import Normalize
import numpy

from ...api.geometry import Interval
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .visarray import ComplexArrayComponent, VisualizationArray


class CylindricalColorModel(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, h: float, x: float, y: float) -> tuple[float, float, float]:
        pass


class HSVSaturationColorModel(CylindricalColorModel):

    @property
    def name(self) -> str:
        return 'HSV Saturation'

    def __call__(self, h: float, x: float, y: float) -> tuple[float, float, float]:
        return colorsys.hsv_to_rgb(h, x, y)


class HSVValueColorModel(CylindricalColorModel):

    @property
    def name(self) -> str:
        return 'HSV Value'

    def __call__(self, h: float, x: float, y: float) -> tuple[float, float, float]:
        return colorsys.hsv_to_rgb(h, y, x)


class HLSLightnessColorModel(CylindricalColorModel):

    @property
    def name(self) -> str:
        return 'HLS Lightness'

    def __call__(self, h: float, x: float, y: float) -> tuple[float, float, float]:
        return colorsys.hls_to_rgb(h, x, y)


class HLSSaturationColorModel(CylindricalColorModel):

    @property
    def name(self) -> str:
        return 'HLS Saturation'

    def __call__(self, h: float, x: float, y: float) -> tuple[float, float, float]:
        return colorsys.hls_to_rgb(h, y, x)


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, arrayComponent: ComplexArrayComponent, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation],
                 variantChooser: PluginChooser[CylindricalColorModel]) -> None:
        super().__init__(arrayComponent, displayRange, transformChooser)
        self._variantChooser = variantChooser
        self._variantChooser.addObserver(self)

    @classmethod
    def createColorizerList(
            cls, array: VisualizationArray, displayRange: DisplayRange,
            transformChooser: PluginChooser[ScalarTransformation]) -> list[Colorizer]:
        arrayComponent = ComplexArrayComponent(array)
        modelList = [
            HSVSaturationColorModel(),
            HSVValueColorModel(),
            HLSLightnessColorModel(),
            HLSSaturationColorModel()
        ]
        variantList = [
            PluginEntry[CylindricalColorModel](simpleName=model.name,
                                               displayName=model.name,
                                               strategy=model) for model in modelList
        ]
        variantChooser = PluginChooser[CylindricalColorModel].createFromList(variantList)
        return [cls(arrayComponent, displayRange, transformChooser, variantChooser)]

    def getVariantNameList(self) -> list[str]:
        return self._variantChooser.getDisplayNameList()

    def getVariantName(self) -> str:
        return self._variantChooser.getCurrentDisplayName()

    def setVariantByName(self, name: str) -> None:
        self._variantChooser.setFromDisplayName(name)

    def getDataRange(self) -> Interval[Decimal]:
        values = numpy.absolute(self._arrayComponent())
        lower = Decimal(repr(values.min()))
        upper = Decimal(repr(values.max()))
        return Interval[Decimal](lower, upper)

    def __call__(self) -> RealArrayType:
        if self._displayRange.getUpper() <= self._displayRange.getLower():
            shape = self._arrayComponent().shape
            return numpy.zeros((*shape, 4))

        amplitude = numpy.absolute(self._arrayComponent())
        phaseInRadians = numpy.angle(self._arrayComponent())
        norm = Normalize(vmin=float(self._displayRange.getLower()),
                         vmax=float(self._displayRange.getUpper()),
                         clip=False)

        h = (phaseInRadians + numpy.pi) / (2 * numpy.pi)
        x = norm(amplitude)
        y = numpy.ones_like(h)
        a = numpy.ones_like(h)

        model = numpy.vectorize(self._variantChooser.getCurrentStrategy())
        r, g, b = model(h, x, y)

        return numpy.stack((r, g, b, a), axis=-1)

    def update(self, observable: Observable) -> None:
        if observable is self._variantChooser:
            self.notifyObservers()
        else:
            super().update(observable)
