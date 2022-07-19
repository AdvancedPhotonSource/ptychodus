from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import colorsys
import logging

import matplotlib.colors
import matplotlib.pyplot
import numpy
import numpy.typing

from ..api.geometry import Interval
from ..api.image import RealArrayType, ScalarTransformation
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageExtent:
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def __add__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width + other.width
            h = self.height + other.height
            return ImageExtent(width=w, height=h)

    def __sub__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width - other.width
            h = self.height - other.height
            return ImageExtent(width=w, height=h)

    def __mul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width * other
            h = self.height * other
            return ImageExtent(width=w, height=h)

    def __rmul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = other * self.width
            h = other * self.height
            return ImageExtent(width=w, height=h)

    def __floordiv__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width // other
            h = self.height // other
            return ImageExtent(width=w, height=h)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.width}, {self.height})'


class NumericArrayComponents(Observable, Observer):
    def __init__(self, scalarTransformationChooser: PluginChooser[ScalarTransformation]) -> None:
        super().__init__(self)
        self._scalarTransformationChooser = scalarTransformationChooser
        self._array = numpy.empty((0,0))

    @classmethod
    def createInstance(cls, scalarTransformationChooser: PluginChooser[ScalarTransformation]) -> NumericArrayComponents:
        components = cls(scalarTransformationChooser)
        scalarTransformationChooser.addObserver(components)
        return components

    def setArray(self, array: numpy.typing.NDArray[numpy.number]) -> None:
        if array.dtype == numpy.inexact:
            self._array = array
        elif array.dtype == numpy.integer:
            self._array = array.astype(numpy.float64)
        else:
            logger.error(f'Refusing to assign array with non-numeric dtype \"{array.dtype}\"!')
            self._array = numpy.empty((0,0))

        self.notifyObservers()

    def getScalarTransformationList(self) -> list[str]:
        return self._scalarTransformationChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self._scalarTransformationChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._scalarTransformationChooser.setFromDisplayName(name)

    def _transform(self, array: RealArrayType) -> RealArrayType:
        scalarTransform = self._scalarTransformationChooser.getCurrentStrategy()
        return scalarTransform(array)

    @property
    def real(self) -> RealArrayType:
        return numpy.real(self._array)

    @property
    def realTransformed(self) -> RealArrayType:
        return self._transform(self.real)

    @property
    def imaginary(self) -> RealArrayType:
        return numpy.imag(self._array)

    @property
    def imaginaryTransformed(self) -> RealArrayType:
        return self._transform(self.imaginary)

    @property
    def amplitude(self) -> RealArrayType:
        return numpy.absolute(self._array)

    @property
    def amplitudeTransformed(self) -> RealArrayType:
        return self._transform(self.amplitude)

    @property
    def phaseInRadians(self) -> RealArrayType:
        return numpy.angle(self._array)

    @property
    def phaseInRadiansTransformed(self) -> RealArrayType:
        return self._transform(self.phaseInRadians)

    @property
    def phaseNormalized(self) -> RealArrayType:
        return (self.phaseInRadians + numpy.pi) / (2 * numpy.pi)

    @property
    def zeros(self) -> RealArrayType:
        return numpy.zeros_like(self.real)

    def update(self, observable: Observable) -> None:
        if observable is self._scalarTransformationChooser:
            self.notifyObservers()


class Colorizer(Callable[[NumericArrayComponents], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass


class AmplitudeInSaturationHSVColorizer(Colorizer):

    @property
    def name(self) -> str:
        return 'HSV Saturation'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        f = numpy.vectorize(colorsys.hsv_to_rgb)
        h = components.phaseNormalized
        s = components.amplitudeTransformed
        v = components.zeros
        return numpy.stack(f(h, s, v), axis=-1)


class AmplitudeInValueHSVColorizer(Colorizer):

    @property
    def name(self) -> str:
        return 'HSV Value'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        f = numpy.vectorize(colorsys.hsv_to_rgb)
        h = components.phaseNormalized
        s = components.zeros
        v = components.amplitudeTransformed
        return numpy.stack(f(h, s, v), axis=-1)


class AmplitudeInLightnessHLSColorizer(Colorizer):

    @property
    def name(self) -> str:
        return 'HLS Lightness'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        f = numpy.vectorize(colorsys.hls_to_rgb)
        h = components.phaseNormalized
        l = components.amplitudeTransformed
        s = components.zeros
        return numpy.stack(f(h, l, s), axis=-1)


class AmplitudeInSaturationHLSColorizer(Colorizer):

    @property
    def name(self) -> str:
        return 'HLS Saturation'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        f = numpy.vectorize(colorsys.hls_to_rgb)
        h = components.phaseNormalized
        l = components.zeros
        s = components.amplitudeTransformed
        return numpy.stack(f(h, l, s), axis=-1)


class MappedColorizer(Colorizer, Observable, Observer):

    def __init__(self, name: str, function: Callable[[NumericArrayComponents],RealArrayType],
            colormapChooser: PluginChooser[matplotlib.colors.Colormap]) -> None:
        self._name = name
        self._function = function
        self._colormapChooser = colormapChooser

    @classmethod
    def createInstance(cls, name: str, function: Callable[[NumericArrayComponents],RealArrayType],
            colormapChooser: PluginChooser[matplotlib.colors.Colormap]) -> MappedColorizer:
        colorizer = cls(name, function, colormapChooser)
        colormapChooser.addObserver(colorizer)
        # FIXME cyclic/acyclic
        # FIXME notifyObservers when colormap changes
        return colorizer

    @property
    def name(self) -> str:
        return self._name

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


class ComplexAmplitudeStrategy(Colorizer):

    @property
    def name(self) -> str:
        return 'Amplitude'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        return components.amplitudeTransformed


class ComplexPhaseStrategy(Colorizer):

    @property
    def name(self) -> str:
        return 'Phase'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        return components.phaseInRadiansTransformed


class ComplexRealComponentStrategy(Colorizer):

    @property
    def name(self) -> str:
        return 'Real'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        return components.realTransformed


class ComplexImaginaryComponentStrategy(Colorizer):

    @property
    def name(self) -> str:
        return 'Imaginary'

    def __call__(self, components: NumericArrayComponents) -> RealArrayType:
        return components.imaginaryTransformed


class ColormapChooserFactory:

    def __init__(self):
        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        self._cyclicColormapList = ['twilight', 'twilight_shifted', 'hsv']
        self._acyclicColormapList = [cm for cm in matplotlib.pyplot.colormaps() \
                if cm not in self._cyclicColormapList]

    def createCyclicColormapChooser(self) -> PluginChooser[matplotlib.colors.Colormap]:
        return PluginChooser[matplotlib.colors.Colormap].createFromList(
            [self._createEntry(name) for name in self._cyclicColormapList])

    def createAcyclicColormapChooser(self) -> PluginChooser[matplotlib.colors.Colormap]:
        return PluginChooser[matplotlib.colors.Colormap].createFromList(
            [self._createEntry(name) for name in self._acyclicColormapList])

    def _createEntry(self, cmap: str) -> PluginEntry[matplotlib.colors.Colormap]:
        return PluginEntry[matplotlib.colors.Colormap](simpleName=cmap,
                                                       displayName=cmap,
                                                       strategy=matplotlib.cm.get_cmap(cmap))


class ImagePresenter(Observable, Observer):

    def __init__(self, colormapChooserFactory: ColormapChooserFactory,
                 colorizerChooser: PluginChooser[Colorizer]) -> None:
        super().__init__()
        self._cyclicColormapChooser = colormapChooserFactory.createCyclicColormapChooser()
        self._cyclicColormapChooser.setFromSimpleName('hsv')
        self._acyclicColormapChooser = colormapChooserFactory.createAcyclicColormapChooser()
        self._acyclicColormapChooser.setFromSimpleName('viridis')
        self._colormapChooser = self._acyclicColormapChooser
        self._colorizerChooser = colorizerChooser
        self._array: Optional[numpy.typing.NDArray] = None
        self._image: Optional[numpy.typing.NDArray] = None
        self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRangeLimits = Interval[Decimal](Decimal(0), Decimal(1))

    @classmethod
    def createInstance(
            cls, colormapChooserFactory: ColormapChooserFactory,
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
        return self._scalarTransformationChooser.getDisplayNameList() # FIXME

    def getScalarTransformation(self) -> str:
        return self._scalarTransformationChooser.getCurrentDisplayName() # FIXME

    def setScalarTransformation(self, name: str) -> None:
        self._scalarTransformationChooser.setFromDisplayName(name) # FIXME

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

    # FIXME do this via dependency injection
    def setArray(self, array: numpy.typing.NDArray) -> None:
        self._array = array
        self._updateImage()
        self.notifyObservers()

    def _updateImage(self) -> None:
        if self._array is None:
            self._image = None
        else:
            colorizer = self._colorizerChooser.getCurrentStrategy()
            scalarTransform = self._scalarTransformationChooser.getCurrentStrategy()

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

    def _updateColormap(self) -> None:
        isCyclic = self._colorizerChooser.getCurrentStrategy().isCyclic
        self._colormapChooser = self._cyclicColormapChooser if isCyclic else self._acyclicColormapChooser

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

    def isComplexValued(self) -> bool:
        return False if self._array is None else numpy.iscomplexobj(self._array)

    def update(self, observable: Observable) -> None:
        if observable is self._colormapChooser:
            self.notifyObservers()
        elif observable is self._scalarTransformationChooser:
            self._updateImage()
            self.notifyObservers()
        elif observable is self._colorizerChooser:
            self._updateImage()
            self._updateColormap()
            self.notifyObservers()
