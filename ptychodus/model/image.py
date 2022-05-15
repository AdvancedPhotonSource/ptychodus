from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Tuple

import matplotlib.pyplot
import numpy
import numpy.typing

from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry

# TODO encode magnitude/phase in rgb/hsv channels


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
        self._colormapChooser = self._cyclicColormapChooser if complexToRealStrategyChooser.getCurrentStrategy(
        ).isCyclic else self._acyclicColormapChooser
        self._scalarTransformationChooser = scalarTransformationChooser
        self._complexToRealStrategyChooser = complexToRealStrategyChooser
        self._array: Optional[numpy.typing.NDArray] = None
        self._image: Optional[numpy.typing.NDArray] = None
        self._vminIsAuto = False
        self._vmin = Decimal(0)
        self._vmaxIsAuto = False
        self._vmax = Decimal(1)

    @classmethod
    def createInstance(cls, colormapChooserFactory: ColormapChooserFactory,
                       scalarTransformationChooser: PluginChooser[ScalarTransformation],
                       complexToRealStrategyChooser: PluginChooser[ComplexToRealStrategy]) -> None:
        presenter = cls(colormapChooserFactory, scalarTransformationChooser,
                        complexToRealStrategyChooser)
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
        isCyclic = self._complexToRealStrategyChooser.getCurrentStrategy().isCyclic
        self._colormapChooser = self._cyclicColormapChooser if isCyclic else self._acyclicColormapChooser

    def isAutomaticVMinEnabled(self) -> bool:
        return self._vminIsAuto

    def setAutomaticVMinEnabled(self, enabled: bool) -> None:
        self._vminIsAuto = enabled
        self._updateImageAndNotifyObservers()

    def getVMinValue(self) -> Decimal:
        return self._vmin

    def setVMinValue(self, vmin: Decimal) -> None:
        self._vmin = vmin
        self.notifyObservers()

    def isAutomaticVMaxEnabled(self) -> bool:
        return self._vmaxIsAuto

    def setAutomaticVMaxEnabled(self, enabled: bool) -> None:
        self._vmaxIsAuto = enabled
        self._updateImageAndNotifyObservers()

    def getVMaxValue(self) -> Decimal:
        return self._vmax

    def setVMaxValue(self, vmax: Decimal) -> None:
        self._vmax = vmax
        self.notifyObservers()

    # TODO do this via dependency injection
    def setArray(self, array: numpy.typing.NDArray) -> None:
        self._array = array
        self._updateImageAndNotifyObservers()

    def _updateImageAndNotifyObservers(self) -> None:
        if self._array is None:
            self._image = None
            return

        complexToRealStrategy = self._complexToRealStrategyChooser.getCurrentStrategy()
        realArray = complexToRealStrategy(self._array) if numpy.iscomplexobj(
            self._array) else self._array

        scalarTransform = self._scalarTransformationChooser.getCurrentStrategy()
        self._image = scalarTransform(realArray)

        if self._vminIsAuto:
            self._vmin = Decimal(repr(self._image.min()))

        if self._vmaxIsAuto:
            self._vmax = Decimal(repr(self._image.max()))

        self.notifyObservers()

    def getImage(self) -> Optional[numpy.typing.NDArray]:
        if self._image is None:
            return

        if self._vmax <= self._vmin:
            return

        cnorm = matplotlib.colors.Normalize(vmin=self._vmin, vmax=self._vmax, clip=False)
        cmap = self._colormapChooser.getCurrentStrategy()
        scalarMappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

        return scalarMappable.to_rgba(self._image)

    def update(self, observable: Observable) -> None:
        if observable is self._cyclicColormapChooser:
            self.notifyObservers()
        elif observable is self._acyclicColormapChooser:
            self.notifyObservers()
        elif observable is self._scalarTransformationChooser:
            self._updateImageAndNotifyObservers()
        elif observable is self._complexToRealStrategyChooser:
            self._updateImageAndNotifyObservers()
