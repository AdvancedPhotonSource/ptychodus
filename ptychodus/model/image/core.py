from __future__ import annotations
from decimal import Decimal
from typing import Optional
import logging

import numpy
import numpy.typing

from ...api.geometry import Interval
from ...api.image import RealArrayType
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .mappedColorizer import MappedColorizer
from .visarray import NumericArrayType, VisualizationArray

logger = logging.getLogger(__name__)


class ImagePresenter(Observable, Observer):

    def __init__(self, array: VisualizationArray,
                 colorizerChooser: PluginChooser[Colorizer]) -> None:
        super().__init__()
        self._array = array
        self._colorizerChooser = colorizerChooser
        self._image: Optional[RealArrayType] = None
        self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRange = Interval[Decimal](Decimal(0), Decimal(1))
        self._displayRangeLimits = Interval[Decimal](Decimal(0), Decimal(1))

    @classmethod
    def createInstance(cls, array: VisualizationArray,
                       colorizerChooser: PluginChooser[Colorizer]) -> ImagePresenter:
        presenter = cls(array, colorizerChooser)
        colorizerChooser.addObserver(presenter)
        return presenter

    # TODO do this via dependency injection
    def setArray(self, array: NumericArrayType) -> None:
        self._array.setArray(array)

    def getColorizerList(self) -> list[str]:
        return self._colorizerChooser.getDisplayNameList()

    def getColorizer(self) -> str:
        return self._colorizerChooser.getCurrentDisplayName()

    def setColorizer(self, name: str) -> None:
        self._colorizerChooser.setFromDisplayName(name)

    @property
    def _colorizer(self) -> Colorizer:
        return self._colorizerChooser.getCurrentStrategy()

    def getScalarTransformationList(self) -> list[str]:
        return self._colorizer.getScalarTransformationList()

    def getScalarTransformation(self) -> str:
        return self._colorizer.getScalarTransformation()

    def setScalarTransformation(self, name: str) -> None:
        self._colorizer.setScalarTransformation(name)

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

    def isColormapEnabled(self) -> bool:
        return isinstance(self._colorizer, MappedColorizer)

    def getColormapList(self) -> list[str]:
        colormapList: list[str] = list()

        if isinstance(self._colorizer, MappedColorizer):
            colormapList.extend(self._colorizer.getColormapList())

        return colormapList

    def getColormap(self) -> str:
        colormap = str()

        if isinstance(self._colorizer, MappedColorizer):
            colormap = self._colorizer.getColormap()

        return colormap

    def setColormap(self, name: str) -> None:
        if isinstance(self._colorizer, MappedColorizer):
            self._colorizer.setColormap(name)
        else:
            logger.error('Colorizer does not accept a colormap.')

    def _updateImage(self) -> None:  # FIXME
        if self._image is None or numpy.size(self._image) <= 0:
            self._dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        else:
            vmin = Decimal(repr(self._image.min()))
            vmax = Decimal(repr(self._image.max()))

            if vmin == vmax:
                vmax += 1

            self._dataRange = Interval[Decimal](vmin, vmax)

        self.notifyObservers()

    def getImage(self) -> Optional[numpy.typing.NDArray]:
        return self._image

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            self._updateImage()


class ImageCore:
    pass  # FIXME
