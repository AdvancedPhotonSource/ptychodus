from __future__ import annotations
from collections.abc import Iterator, Sequence
from typing import Final
import logging

import numpy

from ...api.geometry import Interval, Line2D
from ...api.observer import Observable, Observer
from ...api.patterns import PixelGeometry
from ...api.plugins import PluginChooser
from ...api.visualization import LineCut, RealArrayType, ScalarTransformation
from .renderer import Renderer
from .displayRange import DisplayRange
from .mappedRenderer import MappedRenderer
from .modelRenderer import CylindricalColorModelRenderer
from .visarray import NumberArrayType, VisualizationArray

logger = logging.getLogger(__name__)


class ImagePresenter(Observable, Observer):
    EPS: Final[float] = 1.e-6

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 rendererChooser: PluginChooser[Renderer]) -> None:
        super().__init__()
        self._array = array
        self._displayRange = displayRange
        self._rendererChooser = rendererChooser
        self._image = rendererChooser.currentPlugin.strategy()

    @classmethod
    def createInstance(cls, array: VisualizationArray, displayRange: DisplayRange,
                       rendererChooser: PluginChooser[Renderer]) -> ImagePresenter:
        presenter = cls(array, displayRange, rendererChooser)
        displayRange.addObserver(presenter)
        rendererChooser.addObserver(presenter)
        rendererChooser.currentPlugin.strategy.addObserver(presenter)
        return presenter

    def setArray(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> None:
        self._array.setArray(array, pixelGeometry)

    def clearArray(self) -> None:
        self._array.clearArray()

    @property
    def _renderer(self) -> Renderer:
        return self._rendererChooser.currentPlugin.strategy

    def getRendererNameList(self) -> Sequence[str]:
        return self._rendererChooser.getDisplayNameList()

    def getRendererName(self) -> str:
        return self._rendererChooser.currentPlugin.displayName

    def setRendererByName(self, name: str) -> None:
        self._renderer.removeObserver(self)
        self._rendererChooser.setCurrentPluginByName(name)
        self._renderer.addObserver(self)

    def getColorSamples(self, normalizedValues: RealArrayType) -> RealArrayType:
        renderer = self._rendererChooser.currentPlugin.strategy
        return renderer.getColorSamples(normalizedValues)

    def isRendererCyclic(self) -> bool:
        renderer = self._rendererChooser.currentPlugin.strategy
        return renderer.isCyclic()

    def getScalarTransformationNameList(self) -> Sequence[str]:
        return self._renderer.getScalarTransformationNameList()

    def getScalarTransformationName(self) -> str:
        return self._renderer.getScalarTransformationName()

    def setScalarTransformationByName(self, name: str) -> None:
        self._renderer.setScalarTransformationByName(name)

    def getDisplayRangeLimits(self) -> Interval[float]:
        return self._displayRange.getLimits()

    def getMinDisplayValue(self) -> float:
        return self._displayRange.getLower()

    def setMinDisplayValue(self, value: float) -> None:
        self._displayRange.setLower(value)

    def getMaxDisplayValue(self) -> float:
        return self._displayRange.getUpper()

    def setMaxDisplayValue(self, value: float) -> None:
        self._displayRange.setUpper(value)

    def setDisplayRangeToDataRange(self) -> None:
        dataRange = Interval[float](0., 1.)
        values = self._renderer.getDataArray()

        if numpy.size(values) > 0:
            dataRange = Interval[float](values.min(), values.max())

            if numpy.isnan(dataRange.lower) or numpy.isnan(dataRange.upper):
                logger.debug('Visualization array component includes one or more NaNs.')
                dataRange = Interval[float](0., 1.)
            elif dataRange.lower == dataRange.upper:
                logger.debug('Visualization array component values are uniform.')
                dataRange.lower -= 0.5
                dataRange.upper += 0.5

        self._displayRange.setRangeAndLimits(dataRange)

    def setCustomDisplayRange(self, minValue: float, maxValue: float) -> None:
        self._displayRange.setLimits(minValue, maxValue)

    def getVariantNameList(self) -> Sequence[str]:
        return self._renderer.getVariantNameList()

    def getVariantName(self) -> str:
        return self._renderer.getVariantName()

    def setVariantByName(self, name: str) -> None:
        self._renderer.setVariantByName(name)

    def getImage(self) -> RealArrayType:
        return self._image

    def _updateImage(self) -> None:
        try:
            image = self._renderer()
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

    def _clipToBoundingBox(self, line: Line2D) -> Interval[float]:
        arrayShape = self._array.shape
        alphaX = self._intersectBoundingBox(line.begin.x, line.end.x, arrayShape[-1])
        alphaY = self._intersectBoundingBox(line.begin.y, line.end.y, arrayShape[-2])
        return Interval[float].createProper(
            max(0., max(alphaX.lower, alphaY.lower)),
            min(1., min(alphaX.upper, alphaY.upper)),
        )

    def _intersectGrid(self, line: Line2D) -> Sequence[float]:
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

    def getLineCut(self, line: Line2D) -> LineCut:
        intersections = self._intersectGrid(line)
        dataLabel = self._renderer.getDataLabel()
        dataArray = self._renderer.getDataArray()

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
        if observable is self._rendererChooser:
            self._updateImage()
        elif observable is self._renderer:
            self._updateImage()


class ImageCore:

    def __init__(self, transformChooser: PluginChooser[ScalarTransformation], *,
                 isComplex: bool) -> None:
        self._array = VisualizationArray()
        self._displayRange = DisplayRange()
        self._rendererChooser = PluginChooser[Renderer]()

        cargs = (self._array, self._displayRange, transformChooser)

        for renderer in CylindricalColorModelRenderer.createRendererVariants(*cargs,
                                                                             isComplex=isComplex):
            self._rendererChooser.registerPlugin(renderer, displayName=renderer.name)

        for renderer in MappedRenderer.createRendererVariants(*cargs, isComplex=isComplex):
            self._rendererChooser.registerPlugin(renderer, displayName=renderer.name)

        self.presenter = ImagePresenter.createInstance(self._array, self._displayRange,
                                                       self._rendererChooser)
