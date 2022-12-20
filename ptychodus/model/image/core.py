from __future__ import annotations
from decimal import Decimal
import logging

from ...api.geometry import Interval
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .mappedColorizer import MappedColorizer
from .modelColorizer import CylindricalColorModelColorizer
from .visarray import NumericArrayType, VisualizationArray

logger = logging.getLogger(__name__)


class ImagePresenter(Observable, Observer):

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 colorizerChooser: PluginChooser[Colorizer]) -> None:
        super().__init__()
        self._array = array
        self._displayRange = displayRange
        self._colorizerChooser = colorizerChooser
        self._image = colorizerChooser.getCurrentStrategy()()

    @classmethod
    def createInstance(cls, array: VisualizationArray, displayRange: DisplayRange,
                       colorizerChooser: PluginChooser[Colorizer]) -> ImagePresenter:
        presenter = cls(array, displayRange, colorizerChooser)
        displayRange.addObserver(presenter)
        colorizerChooser.addObserver(presenter)
        colorizerChooser.getCurrentStrategy().addObserver(presenter)
        return presenter

    # TODO do this via dependency injection
    def setArray(self, array: NumericArrayType) -> None:
        self._array.setArray(array)

    @property
    def _colorizer(self) -> Colorizer:
        return self._colorizerChooser.getCurrentStrategy()

    def getColorizerNameList(self) -> list[str]:
        return self._colorizerChooser.getDisplayNameList()

    def getColorizerName(self) -> str:
        return self._colorizerChooser.getCurrentDisplayName()

    def setColorizerByName(self, name: str) -> None:
        self._colorizer.removeObserver(self)
        self._colorizerChooser.setFromDisplayName(name)
        self._colorizer.addObserver(self)

    def getScalarTransformationNameList(self) -> list[str]:
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
        values = self._colorizer.getDataArray()
        lower = Decimal(repr(values.min()))
        upper = Decimal(repr(values.max()))
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

    def getVariantNameList(self) -> list[str]:
        return self._colorizer.getVariantNameList()

    def getVariantName(self) -> str:
        return self._colorizer.getVariantName()

    def setVariantByName(self, name: str) -> None:
        self._colorizer.setVariantByName(name)

    def getImage(self) -> RealArrayType:
        return self._image

    def _updateImage(self) -> None:
        self._image = self._colorizer()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            self._updateImage()
        elif observable is self._colorizer:
            self._updateImage()


class ImageCore:

    @staticmethod
    def _createColorizerPlugin(colorizer: Colorizer) -> PluginEntry[Colorizer]:
        return PluginEntry[Colorizer](simpleName=colorizer.name,
                                      displayName=colorizer.name,
                                      strategy=colorizer)

    def __init__(self, transformChooser: PluginChooser[ScalarTransformation]) -> None:
        self._array = VisualizationArray()
        self._displayRange = DisplayRange()

        colorizerArgs = (self._array, self._displayRange, transformChooser)
        colorizerList = CylindricalColorModelColorizer.createColorizerList(*colorizerArgs) \
                + MappedColorizer.createColorizerList(*colorizerArgs)
        self._colorizerChooser = PluginChooser[Colorizer].createFromList(
            [ImageCore._createColorizerPlugin(colorizer) for colorizer in colorizerList])
        self.presenter = ImagePresenter.createInstance(self._array, self._displayRange,
                                                       self._colorizerChooser)
