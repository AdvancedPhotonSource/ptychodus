from __future__ import annotations
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Final
import logging

import numpy

from ...api.apparatus import PixelGeometry
from ...api.geometry import Interval, Line2D
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plot import LineCut
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .mappedColorizer import MappedColorizer
from .modelColorizer import CylindricalColorModelColorizer
from .visarray import NumericArrayType, VisualizationArray

logger = logging.getLogger(__name__)


class ImagePresenter(Observable, Observer):
    EPS: Final[float] = 1.e-6

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 colorizerChooser: PluginChooser[Colorizer]) -> None:
        super().__init__()
        self._array = array
        self._displayRange = displayRange
        self._colorizerChooser = colorizerChooser
        self._image = colorizerChooser.currentPlugin.strategy()

    @classmethod
    def createInstance(cls, array: VisualizationArray, displayRange: DisplayRange,
                       colorizerChooser: PluginChooser[Colorizer]) -> ImagePresenter:
        presenter = cls(array, displayRange, colorizerChooser)
        displayRange.addObserver(presenter)
        colorizerChooser.addObserver(presenter)
        colorizerChooser.currentPlugin.strategy.addObserver(presenter)
        return presenter

    def setArray(self, array: NumericArrayType, pixelGeometry: PixelGeometry) -> None:
        self._array.setArray(array, pixelGeometry)

    def clearArray(self) -> None:
        self._array.clearArray()

    @property
    def _colorizer(self) -> Colorizer:
        return self._colorizerChooser.currentPlugin.strategy

    def getColorizerNameList(self) -> Sequence[str]:
        return self._colorizerChooser.getDisplayNameList()

    def getColorizerName(self) -> str:
        return self._colorizerChooser.currentPlugin.displayName

    def setColorizerByName(self, name: str) -> None:
        self._colorizer.removeObserver(self)
        self._colorizerChooser.setCurrentPluginByName(name)
        self._colorizer.addObserver(self)

    def getColorSamples(self, normalizedValues: RealArrayType) -> RealArrayType:
        colorizer = self._colorizerChooser.currentPlugin.strategy
        return colorizer.getColorSamples(normalizedValues)

    def isColorizerCyclic(self) -> bool:
        colorizer = self._colorizerChooser.currentPlugin.strategy
        return colorizer.isCyclic()

    def getScalarTransformationNameList(self) -> Sequence[str]:
        return self._colorizer.getScalarTransformationNameList()

    def getScalarTransformationName(self) -> str:
        return self._colorizer.getScalarTransformationName()

    def setScalarTransformationByName(self, name: str) -> None:
        self._colorizer.setScalarTransformationByName(name)

    def getDisplayRangeLimits(self) -> Interval[Decimal]:
        return self._displayRange.getLimits()

    def getMinDisplayValue(self) -> Decimal:
        return self._displayRange.getLower()

    def setMinDisplayValue(self, value: Decimal) -> None:
        self._displayRange.setLower(value)

    def getMaxDisplayValue(self) -> Decimal:
        return self._displayRange.getUpper()

    def setMaxDisplayValue(self, value: Decimal) -> None:
        self._displayRange.setUpper(value)

    def setDisplayRangeToDataRange(self) -> None:
        dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        values = self._colorizer.getDataArray()

        if numpy.size(values) > 0:
            lower = Decimal.from_float(values.min())
            upper = Decimal.from_float(values.max())
            dataRange = Interval[Decimal](lower, upper)

            if dataRange.lower.is_nan() or dataRange.upper.is_nan():
                logger.debug('Visualization array component includes one or more NaNs.')
                dataRange = Interval[Decimal](Decimal(0), Decimal(1))
            elif dataRange.lower == dataRange.upper:
                logger.debug('Visualization array component values are uniform.')
                half = Decimal('0.5')
                dataRange.lower -= half
                dataRange.upper += half

        self._displayRange.setRangeAndLimits(dataRange)

    def setCustomDisplayRange(self, minValue: Decimal, maxValue: Decimal) -> None:
        self._displayRange.setLimits(minValue, maxValue)

    def getVariantNameList(self) -> Sequence[str]:
        return self._colorizer.getVariantNameList()

    def getVariantName(self) -> str:
        return self._colorizer.getVariantName()

    def setVariantByName(self, name: str) -> None:
        self._colorizer.setVariantByName(name)

    def getImage(self) -> RealArrayType:
        return self._image

    def _updateImage(self) -> None:
        try:
            image = self._colorizer()
        except Exception:
            logger.exception('Failed to render image!')
            return

        self._image = image
        self.notifyObservers()

    @staticmethod
    def _intersectBoundingBox(begin: float, end: float, n: int) -> Interval[float]:
        length = end - begin

        if abs(length) < ImagePresenter.EPS:
            return Interval[float](-numpy.inf, numpy.inf)
        else:
            return Interval[float].createProper(
                (0 - begin) / length,
                (n - begin) / length,
            )

    @staticmethod
    def _intersectGridLines(begin: float, end: float,
                            alphaLimits: Interval[float]) -> Iterator[float]:
        ibegin = int(begin)
        iend = int(end)

        if iend < ibegin:
            ibegin, iend = iend, ibegin

        length = end - begin

        if abs(length) > ImagePresenter.EPS:
            for idx in range(ibegin, iend + 1):
                alpha = (idx - begin) / length

                if alpha in alphaLimits:
                    yield alpha

    def _clipToBoundingBox(self, line: Line2D[float]) -> Interval[float]:
        arrayShape = self._array.shape
        alphaX = self._intersectBoundingBox(line.begin.x, line.end.x, arrayShape[-1])
        alphaY = self._intersectBoundingBox(line.begin.y, line.end.y, arrayShape[-2])
        return Interval[float].createProper(
            max(0., max(alphaX.lower, alphaY.lower)),
            min(1., min(alphaX.upper, alphaY.upper)),
        )

    def _intersectGrid(self, line: Line2D[float]) -> Sequence[float]:
        alphaLimits = self._clipToBoundingBox(line)
        xIntersections = [
            x for x in self._intersectGridLines(line.begin.x, line.end.x, alphaLimits)
        ]
        yIntersections = [
            x for x in self._intersectGridLines(line.begin.y, line.end.y, alphaLimits)
        ]

        alpha = {alphaLimits.lower, alphaLimits.upper}
        alpha = alpha.union(xIntersections)
        alpha = alpha.union(yIntersections)
        return sorted(alpha)

    def getLineCut(self, line: Line2D[float]) -> LineCut:
        intersections = self._intersectGrid(line)
        dataLabel = self._colorizer.getDataLabel()
        dataArray = self._colorizer.getDataArray()

        pixelGeometry = self._array.pixelGeometry
        dx = (line.end.x - line.begin.x) * pixelGeometry.widthInMeters
        dy = (line.end.y - line.begin.y) * pixelGeometry.heightInMeters
        lineLength = numpy.hypot(dx, dy)

        distances: list[float] = list()
        values: list[float] = list()

        for alphaL, alphaR in zip(intersections[:-1], intersections[1:]):
            alpha = (alphaL + alphaR) / 2.
            point = line.lerp(alpha)
            value = dataArray[int(point.y), int(point.x)]

            distances.append(alpha * lineLength)
            values.append(value)

        return LineCut(distances, values, dataLabel)

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            self._updateImage()
        elif observable is self._colorizer:
            self._updateImage()


class ImageCore:

    def __init__(self, transformChooser: PluginChooser[ScalarTransformation], *,
                 isComplex: bool) -> None:
        self._array = VisualizationArray()
        self._displayRange = DisplayRange()
        self._colorizerChooser = PluginChooser[Colorizer]()

        cargs = (self._array, self._displayRange, transformChooser)

        for colorizer in CylindricalColorModelColorizer.createColorizerVariants(
                *cargs, isComplex=isComplex):
            self._colorizerChooser.registerPlugin(colorizer, simpleName=colorizer.name)

        for colorizer in MappedColorizer.createColorizerVariants(*cargs, isComplex=isComplex):
            self._colorizerChooser.registerPlugin(colorizer, simpleName=colorizer.name)

        self.presenter = ImagePresenter.createInstance(self._array, self._displayRange,
                                                       self._colorizerChooser)
