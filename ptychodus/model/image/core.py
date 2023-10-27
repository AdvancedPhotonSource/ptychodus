from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
import logging

from ...api.geometry import Interval, Point2D
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .mappedColorizer import MappedColorizer
from .modelColorizer import CylindricalColorModelColorizer
from .visarray import NumericArrayType, VisualizationArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineCut:
    distance: Sequence[float]
    value: Sequence[float]


class ImagePresenter(Observable, Observer):

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

    def setArray(self, array: NumericArrayType) -> None:
        self._array.setArray(array)

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
        values = self._colorizer.getTransformedDataArray()
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
    def _intersect(x0: float, x1: float, dx: float, n: int) -> Interval[float]:
        eps = 1.e-6
        delta = x1 - x0

        if abs(delta) < eps * dx:
            return Interval[float](0., 1.)
        else:
            alpha0 = (0 * dx - x0) / delta
            alphaN = (n * dx - x0) / delta

            if alpha0 < alphaN:
                return Interval[float](alpha0, alphaN)
            else:
                return Interval[float](alphaN, alpha0)

    def getLineCut(self, start: Point2D[float], end: Point2D[float]) -> LineCut:
        pixelGeometry = PixelGeometry()  # FIXME

        # clip to edges of array
        alphaX = self._intersect(start.x, end.x, float(pixelGeometry.widthInMeters),
                                 self._array.shape[-1])
        alphaY = self._intersect(start.y, end.y, float(pixelGeometry.heightInMeters),
                                 self._array.shape[-2])
        alpha = Interval[float](
            lower=max(0., max(alphaX.lower, alphaY.lower)),
            upper=min(1., min(alphaX.upper, alphaY.upper)),
        )

        values = self._colorizer.getDataArray()  # FIXME
        # FIXME calculate alpha values for vertical and horizontal pixel crossings
        distance = [
            0., 1.
        ]  # FIXME distance along line at midpoint of each alpha interval in physical units
        value = [0., 1.]  # FIXME pixel value
        return LineCut(distance, value)

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            self._updateImage()
        elif observable is self._colorizer:
            self._updateImage()


class ImageCore:

    def __init__(self, transformChooser: PluginChooser[ScalarTransformation]) -> None:
        self._array = VisualizationArray()
        self._displayRange = DisplayRange()
        self._colorizerChooser = PluginChooser[Colorizer]()

        cargs = (self._array, self._displayRange, transformChooser)

        for colorizer in CylindricalColorModelColorizer.createColorizerVariants(*cargs):
            self._colorizerChooser.registerPlugin(colorizer, simpleName=colorizer.name)

        for colorizer in MappedColorizer.createColorizerVariants(*cargs):
            self._colorizerChooser.registerPlugin(colorizer, simpleName=colorizer.name)

        self.presenter = ImagePresenter.createInstance(self._array, self._displayRange,
                                                       self._colorizerChooser)
