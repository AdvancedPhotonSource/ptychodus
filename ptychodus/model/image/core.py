from __future__ import annotations
from decimal import Decimal
import logging

from ...api.geometry import Interval
from ...api.image import RealArrayType
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from .colorizer import Colorizer
from .displayRange import DisplayRange
from .mappedColorizer import MappedColorizer
from .modelColorizer import CylindricalColorModelColorizer
from .visarray import *

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
        dataRange = DisplayRange.createUnitInterval()

        if self._image is not None and numpy.size(self._image) > 0:
            lower = Decimal(repr(self._image.min()))
            upper = Decimal(repr(self._image.max()))

            if lower == upper:
                half = Decimal('0.5')
                lower -= half
                upper += half

            dataRange = Interval[Decimal](lower, upper)

        self._displayRange.setRangeAndLimits(dataRange)

    def setCustomDisplayRange(self, minValue: Decimal, maxValue) -> None:
        self._displayRange.setLimits(minValue, maxValue)

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

    def getImage(self) -> RealArrayType:
        return self._image

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            colorizer = self._colorizerChooser.getCurrentStrategy()
            self._image = colorizer()


class ImageCore:

    @staticmethod
    def createComponentChooser(componentList: list[VisualizationArrayComponent]) -> \
            PluginChooser[VisualizationArrayComponent]:
        entryList: list[PluginEntry[VisualizationArrayComponent]] = list()

        for component in componentList:
            entry = PluginEntry[VisualizationArrayComponent](simpleName=component.name,
                                                             displayName=component.name,
                                                             strategy=component)
            entryList.append(entry)

        return PluginChooser[VisualizationArrayComponent].createFromList(entryList)

    @staticmethod
    def createColorizerPlugin(colorizer: Colorizer) -> PluginEntry[Colorizer]:
        return PluginEntry[Colorizer](simpleName=colorizer.name,
                                      displayName=colorizer.name,
                                      strategy=colorizer)

    def __init__(self, transformChooser: PluginChooser[ScalarTransformation]) -> None:
        self._array = VisualizationArray()
        self._displayRange = DisplayRange()

        self._amplitudeChooser = ImageCore.createComponentChooser([
            AmplitudeArrayComponent(self._array, transformChooser),
        ])

        self._componentChooser = ImageCore.createComponentChooser([
            AmplitudeArrayComponent(self._array, transformChooser),
            PhaseArrayComponent(self._array, transformChooser),
            RealArrayComponent(self._array, transformChooser),
            ImaginaryArrayComponent(self._array, transformChooser),
        ])

        self._mappedColorizer = MappedColorizer.createInstance(self._componentChooser,
                                                               self._displayRange)
        self._colorizerChooser = PluginChooser[Colorizer](ImageCore.createColorizerPlugin(
            self._mappedColorizer))

        for colorizer in CylindricalColorModelColorizer.createVariants(
                self._amplitudeChooser, self._displayRange):
            self._colorizerChooser.addStrategy(ImageCore.createColorizerPlugin(colorizer))

        self.presenter = ImagePresenter.createInstance(self._array, self._displayRange,
                                                            self._colorizerChooser)
