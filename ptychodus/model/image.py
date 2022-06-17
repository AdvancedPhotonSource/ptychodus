from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple
import logging

import matplotlib.colors
import matplotlib.pyplot
import numpy
import numpy.typing

from ..api.geometry import Interval
from ..api.image import ScalarTransformation, ComplexToRealStrategy
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImageExtent:
    width: int
    height: int

    @property
    def shape(self) -> Tuple[int, int]:
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


class ColormapChooserFactory:

    def __init__(self):
        # See https://matplotlib.org/stable/gallery/color/colormap_reference.html
        self._cyclicColormapList = ['twilight', 'twilight_shifted', 'hsv']
        self._acyclicColormapList = [
            cm for cm in matplotlib.pyplot.colormaps() if cm not in self._cyclicColormapList
        ]

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
                 scalarTransformationChooser: PluginChooser[ScalarTransformation],
                 complexToRealStrategyChooser: PluginChooser[ComplexToRealStrategy]) -> None:
        super().__init__()
        self._cyclicColormapChooser = colormapChooserFactory.createCyclicColormapChooser()
        self._cyclicColormapChooser.setFromSimpleName('hsv')
        self._acyclicColormapChooser = colormapChooserFactory.createAcyclicColormapChooser()
        self._acyclicColormapChooser.setFromSimpleName('viridis')
        self._colormapChooser = self._acyclicColormapChooser
        self._scalarTransformationChooser = scalarTransformationChooser
        self._complexToRealStrategyChooser = complexToRealStrategyChooser
        self._array: Optional[numpy.typing.NDArray] = None
        self._image: Optional[numpy.typing.NDArray] = None
        self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRangeLimits = Interval[Decimal](Decimal(0), Decimal(1))

    @classmethod
    def createInstance(
            cls, colormapChooserFactory: ColormapChooserFactory,
            scalarTransformationChooser: PluginChooser[ScalarTransformation],
            complexToRealStrategyChooser: PluginChooser[ComplexToRealStrategy]) -> ImagePresenter:
        presenter = cls(colormapChooserFactory, scalarTransformationChooser,
                        complexToRealStrategyChooser)
        presenter._updateColormap()
        presenter._cyclicColormapChooser.addObserver(presenter)
        presenter._acyclicColormapChooser.addObserver(presenter)
        scalarTransformationChooser.addObserver(presenter)
        complexToRealStrategyChooser.addObserver(presenter)
        return presenter

    def getColormapList(self) -> list[str]:
        return self._colormapChooser.getDisplayNameList()

    def getColormap(self) -> str:
        return self._colormapChooser.getCurrentDisplayName()

    def setColormap(self, name: str) -> None:
        self._colormapChooser.setFromDisplayName(name)

    def isColormapEnabled(self) -> bool:
        isColorized = self._complexToRealStrategyChooser.getCurrentStrategy().isColorized
        return (not isColorized)

    def getScalarTransformationList(self) -> list[str]:
        return self._scalarTransformationChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self._scalarTransformationChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._scalarTransformationChooser.setFromDisplayName(name)

    def getComplexToRealStrategyList(self) -> list[str]:
        return self._complexToRealStrategyChooser.getDisplayNameList()

    def getComplexToRealStrategy(self) -> str:
        return self._complexToRealStrategyChooser.getCurrentDisplayName()

    def setComplexToRealStrategy(self, name: str) -> None:
        self._complexToRealStrategyChooser.setFromDisplayName(name)

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
            complexToRealStrategy = self._complexToRealStrategyChooser.getCurrentStrategy()
            scalarTransform = self._scalarTransformationChooser.getCurrentStrategy()

            if numpy.iscomplexobj(self._array):
                self._image = complexToRealStrategy(self._array, scalarTransform)
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
        isCyclic = self._complexToRealStrategyChooser.getCurrentStrategy().isCyclic
        self._colormapChooser = self._cyclicColormapChooser if isCyclic else self._acyclicColormapChooser

    def getImage(self) -> Optional[numpy.typing.NDArray]:
        if self._complexToRealStrategyChooser.getCurrentStrategy().isColorized:
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
        elif observable is self._complexToRealStrategyChooser:
            self._updateImage()
            self._updateColormap()
            self.notifyObservers()
