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
        colorizerChooser.getCurrentStrategy().addObserver(presenter)
        return presenter

    # TODO do this via dependency injection
    def setArray(self, array: NumericArrayType) -> None:
        self._array.setArray(array)

    def getColorizerList(self) -> list[str]:
        return self._colorizerChooser.getDisplayNameList()

    def getColorizer(self) -> str:
        return self._colorizerChooser.getCurrentDisplayName()

    def setColorizer(self, name: str) -> None:
        self._colorizer.removeObserver(self)
        self._colorizerChooser.setFromDisplayName(name)
        self._colorizer.addObserver(self)

    @property
    def _colorizer(self) -> Colorizer:
        return self._colorizerChooser.getCurrentStrategy()

    def getArrayComponentList(self) -> list[str]:
        return self._colorizer.getArrayComponentList()

    def getArrayComponent(self) -> str:
        return self._colorizer.getArrayComponent()

    def setArrayComponent(self, name: str) -> None:
        self._colorizer.setArrayComponent(name)

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
        dataRange = self._colorizer.getDataRange()

        if dataRange.lower.is_nan() or dataRange.upper.is_nan():
            logger.debug('Visualization array component includes one or more NaNs.')
            dataRange = Interval[Decimal](Decimal(0), Decimal(1))
        elif dataRange.lower == dataRange.upper:
            logger.debug('Visualization array component values are uniform.')
            half = Decimal('0.5')
            dataRange.lower -= half
            dataRange.upper += half

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

    def _updateImage(self) -> None:
        colorizer = self._colorizerChooser.getCurrentStrategy()
        self._image = colorizer()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._colorizerChooser:
            self._updateImage()
        elif observable is self._colorizer:
            self._updateImage()


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
