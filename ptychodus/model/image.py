from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import colorsys
import logging

from matpliblib.colors import Colormap
import matplotlib.colors
import matplotlib.pyplot
import numpy
import numpy.typing

from ..api.geometry import Interval
from ..api.image import RealArrayType, ScalarTransformation
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry

logger = logging.getLogger(__name__)


class VisualizationArray(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._array: numpy.typing.NDArray[numpy.inexact] = numpy.empty((0, 0))

    def setArray(self, array: numpy.typing.NDArray[numpy.number]) -> None:
        if array.dtype == numpy.inexact:
            self._array = array
        elif array.dtype == numpy.integer:
            self._array = array.astype(numpy.float64)
        else:
            logger.error(f'Refusing to assign array with non-numeric dtype \"{array.dtype}\"!')
            self._array = numpy.empty((0, 0))

        self.notifyObservers()

    def getAmplitude(self) -> RealArrayType:
        return numpy.absolute(self._array)

    def getPhaseInRadians(self) -> RealArrayType:
        return numpy.angle(self._array)

    def getPhaseNormalized(self) -> RealArrayType:
        return (self.getPhaseInRadians() + numpy.pi) / (2 * numpy.pi)

    def getReal(self) -> RealArrayType:
        return numpy.real(self._array)

    def getImaginary(self) -> RealArrayType:
        return numpy.imag(self._array)

    def getZeros(self) -> RealArrayType:
        return numpy.zeros_like(self.getComponent())


RealArrayCallable = Callable[[], RealArrayType]


class VisualizationArrayComponent(Observable, Observer):

    @staticmethod
    def _createComponentPlugin(name: str, strategy: RealArrayCallable) -> PluginEntry[RealArrayCallable]:
        return PluginEntry[RealArrayCallable](simpleName=name, displayName=name, strategy=strategy)

    def __init__(self, array: VisualizationArray, componentChooser: PluginChooser[RealArrayCallable], transformChooser: PluginChooser[ScalarTransformation]) -> None:
        super().__init__()
        self._array = array
        self._componentChooser = componentChooser
        self._transformChooser = transformChooser

    @classmethod
    def createInstance(cls, array: VisualizationArray. transformChooser: PluginChooser[ScalarTransformation]) -> VisualizationArrayComponent:
        componentPlugins: list[PluginEntry[RealArrayCallable]] = list()
        componentPlugins.append(cls._createComponentPlugin('Amplitude', array.getAmplitude))
        componentPlugins.append(cls._createComponentPlugin('Phase', array.getPhaseInRadians))
        componentPlugins.append(cls._createComponentPlugin('Real', array.getReal))
        componentPlugins.append(cls._createComponentPlugin('Imaginary', array.getImaginary))
        componentChooser = PluginChooser[RealArrayCallable].createFromList(componentPlugins)

        component = cls(array. componentChooser, transformChooser)
        array.addObserver(component)
        componentChooser.addObserver(component)
        transformChooser.addObserver(component)
        return component

    def getArray(self) -> VisualizationArray:
        return self._array

    def getArrayComponentList(self) -> list[str]:
        return self._componentChooser.getDisplayNameList()

    def getArrayComponent(self) -> str:
        return self._componentChooser.getCurrentDisplayName()

    def setArrayComponent(self, name: str) -> None:
        self._componentChooser.setFromDisplayName(name)

    def getScalarTransformationList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    def getComponent(self) -> RealArrayType:
        component = self._componentChooser.getCurrentStrategy()
        transform = self._transformChooser.getCurrentStrategy()
        return transform(component(self._array))

    def update(self, observable: Observable) -> None:
        if observable is self._array:
            self.notifyObservers()
        elif observable is self._componentChooser:
            self.notifyObservers()
        elif observable is self._transformChooser:
            self.notifyObservers()


class Colorizer(Callable[[NumericArrayComponents], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass


CylindricalColorModel = Callable[[float, float, float], tuple[float, float, float]]


class CylindricalColorModelColorizer(Colorizer):

    def __init__(self, name: str, model: CylindricalColorModel, variant: bool) -> None:
        super().__init__()
        self._name = name
        self._model = numpy.vectorize(model)
        self._variant = variant

    @classmethod
    def createHSVSaturationInstance(cls) -> CylindricalColorModelColorizer:
        return cls('HSV Saturation', colorsys.hsv_to_rgb, False)

    @classmethod
    def createHSVValueInstance(cls) -> CylindricalColorModelColorizer:
        return cls('HSV Value', colorsys.hsv_to_rgb, True)

    @classmethod
    def createHLSLightnessInstance(cls) -> CylindricalColorModelColorizer:
        return cls('HLS Lightness', colorsys.hls_to_rgb, False)

    @classmethod
    def createHLSSaturationInstance(cls) -> CylindricalColorModelColorizer:
        return cls('HLS Saturation', colorsys.hls_to_rgb, True)

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        h = components.getPhaseNormalized()
        x = components.getAmplitudeTransformed()
        y = components.getZeros()

        if self._variant:
            y, x = x, y

        return numpy.stack(self._model(h, x, y), axis=-1)


@dataclass(frozen=True)
class ColormapRepository:
    cyclicColormapChooser: PluginChooser[Colormap]
    acyclicColormapChooser: PluginChooser[Colormap]

    @classmethod
    def createInstance(self) -> ColormapRepository:
        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        cyclicColormapNames = ['twilight', 'twilight_shifted', 'hsv']
        cyclicColormapEntries: list[PluginEntry[Colormap]] = list()
        acyclicColormapEntries: list[PluginEntry[Colormap]] = list()

        for name in matplotlib.pyplot.colormaps():
            entry = PluginEntry[Colormap](simpleName=name,
                                          displayName=name,
                                          strategy=matplotlib.cm.get_cmap(name))

            if name in cyclicColormapNames:
                cyclicColormapEntries.append(entry)
            else:
                acyclicColormapEntries.append(entry)

        cyclicColormapChooser = PluginChooser[Colormap].createFromList(cyclicColormapEntries)
        cyclicColormapChooser.setFromSimpleName('hsv')

        acyclicColormapChooser = PluginChooser[Colormap].createFromList(acyclicColormapEntries)
        acyclicColormapChooser.setFromSimpleName('viridis')

        return cls(cyclicColormapChooser, acyclicColormapChooser)


class MappedColorizer(Colorizer, Observable, Observer):

    def __init__(self, arrayComponent: VisualizationArrayComponent,
                 colormapChooser: PluginChooser[Colormap]) -> None:
        self._arrayComponent = arrayComponent
        self._colormapChooser = colormapChooser

    @classmethod
    def createInstance(cls, arrayComponent: VisualizationArrayComponent,
                       colormapChooser: PluginChooser[Colormap]) -> MappedColorizer:
        colorizer = cls(arrayComponent, colormapChooser)
        arrayComponent.addObserver(colorizer)
        colormapChooser.addObserver(colorizer)
        return colorizer

    @property
    def name(self) -> str:
        return 'Colormap'

    def getColormapList(self) -> list[str]:
        return self._colormapChooser.getDisplayNameList()

    def getColormap(self) -> str:
        return self._colormapChooser.getCurrentDisplayName()

    def setColormap(self, name: str) -> None:
        self._colormapChooser.setFromDisplayName(name)

    @property
    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        array = self._function(components)

        # FIXME displayRange

        cnorm = matplotlib.colors.Normalize(vmin=self._displayRange.lower,
                                            vmax=self._displayRange.upper,
                                            clip=False)
        cmap = self._colormapChooser.getCurrentStrategy()
        scalarMappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)
        return scalarMappable.to_rgba(array)

    def update(self, observable: Observable) -> None:
        if observable is self._colormapChooser:
            self.notifyObservers()


class ImagePresenter(Observable, Observer):

    def __init__(self, colormapChooserFactory: ColormapChooserFactory,
                 colorizerChooser: PluginChooser[Colorizer]) -> None:
        super().__init__()
        self._colorizerChooser = colorizerChooser
        self._array: Optional[numpy.typing.NDArray] = None
        self._image: Optional[numpy.typing.NDArray] = None
        self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRangeLimits = Interval[Decimal](Decimal(0), Decimal(1))

    @classmethod
    def createInstance(cls, colormapChooserFactory: ColormapChooserFactory,
                       colorizerChooser: PluginChooser[Colorizer]) -> ImagePresenter:
        presenter = cls(colormapChooserFactory, colorizerChooser)
        presenter._updateColormap()
        presenter._cyclicColormapChooser.addObserver(presenter)
        presenter._acyclicColormapChooser.addObserver(presenter)
        colorizerChooser.addObserver(presenter)
        return presenter

    def getColormapList(self) -> list[str]:
        return self._colormapChooser.getDisplayNameList()

    def getColormap(self) -> str:
        return self._colormapChooser.getCurrentDisplayName()

    def setColormap(self, name: str) -> None:
        self._colormapChooser.setFromDisplayName(name)

    def isColormapEnabled(self) -> bool:
        isColorized = self._colorizerChooser.getCurrentStrategy().isColorized
        return (not isColorized)

    def getScalarTransformationList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()  # FIXME

    def getScalarTransformation(self) -> str:
        return self._transformChooser.getCurrentDisplayName()  # FIXME

    def setScalarTransformation(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)  # FIXME

    def getColorizerList(self) -> list[str]:
        return self._colorizerChooser.getDisplayNameList()

    def getColorizer(self) -> str:
        return self._colorizerChooser.getCurrentDisplayName()

    def setColorizer(self, name: str) -> None:
        self._colorizerChooser.setFromDisplayName(name)

    def getDisplayRangeLimits(self) -> Interval[Decimal]:
        return self._displayRangeLimits

    def getMinDisplayValue(self) -> Decimal:
        limits = self.getDisplayRangeLimits()
        return limits.clamp(self._displayRange.lower)

    def setMinDisplayValue(self, value: Decimal) -> None:
        self._displayRange.lower = value
        self.notifyObservers()

    def getMaxDisplayValue(self) -> Decimal:
        limits = self.getDisplayRangeLimits()
        return limits.clamp(self._displayRange.upper)

    def setMaxDisplayValue(self, value: Decimal) -> None:
        self._displayRange.upper = value
        self.notifyObservers()

    def setDisplayRangeToDataRange(self) -> None:
        self._displayRange = self._dataRange.copy()
        self._displayRangeLimits = self._dataRange.copy()
        self.notifyObservers()

    def setCustomDisplayRange(self, minValue: Decimal, maxValue) -> None:
        self._displayRangeLimits = Interval[Decimal](minValue, maxValue)
        self.notifyObservers()

    # TODO do this via dependency injection
    def setArray(self, array: numpy.typing.NDArray) -> None:
        self._array = array
        self._updateImage()
        self.notifyObservers()

    def _updateImage(self) -> None:
        if self._array is None:
            self._image = None
        else:
            colorizer = self._colorizerChooser.getCurrentStrategy()
            scalarTransform = self._transformChooser.getCurrentStrategy()

            if numpy.iscomplexobj(self._array):
                self._image = colorizer(self._array, scalarTransform)
            else:
                self._image = scalarTransform(self._array.astype(numpy.float32))

        if self._image is None or numpy.size(self._image) <= 0:
            self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        else:
            vmin = Decimal(repr(self._image.min()))
            vmax = Decimal(repr(self._image.max()))

            if vmin == vmax:
                vmax += 1

            self._dataRange = Interval[Decimal](vmin, vmax)

    def getImage(self) -> Optional[numpy.typing.NDArray]:
        if self._colorizerChooser.getCurrentStrategy().isColorized:
            return self._image
        elif self._image is None or self._displayRange.isEmpty:
            return None
        else:
            cnorm = matplotlib.colors.Normalize(vmin=self._displayRange.lower,
                                                vmax=self._displayRange.upper,
                                                clip=False)
            cmap = self._colormapChooser.getCurrentStrategy()
            scalarMappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)
            return scalarMappable.to_rgba(self._image)

    def update(self, observable: Observable) -> None:
        if observable is self._colormapChooser:
            self.notifyObservers()
        elif observable is self._transformChooser:
            self._updateImage()
            self.notifyObservers()
        elif observable is self._colorizerChooser:
            self._updateImage()
            self._updateColormap()
            self.notifyObservers()
