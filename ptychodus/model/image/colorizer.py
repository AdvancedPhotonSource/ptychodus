from abc import ABC, abstractmethod
from collections.abc import Sequence

from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from .displayRange import DisplayRange
from .visarray import VisualizationArray


class Colorizer(Observable, Observer, ABC):

    def __init__(self, array: VisualizationArray, displayRange: DisplayRange,
                 transformChooser: PluginChooser[ScalarTransformation]) -> None:
        super().__init__()
        self._array = array
        self._array.addObserver(self)
        self._displayRange = displayRange
        self._displayRange.addObserver(self)
        self._transformChooser = transformChooser
        self._transformChooser.addObserver(self)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def getScalarTransformationNameList(self) -> Sequence[str]:
        return self._transformChooser.getDisplayNameList()

    def getScalarTransformationName(self) -> str:
        return self._transformChooser.currentPlugin.displayName

    def setScalarTransformationByName(self, name: str) -> None:
        self._transformChooser.setCurrentPluginByName(name)

    @abstractmethod
    def getVariantNameList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getVariantName(self) -> str:
        pass

    @abstractmethod
    def setVariantByName(self, name: str) -> None:
        pass

    @abstractmethod
    def getColorSamples(self, normalizedValues: RealArrayType) -> RealArrayType:
        pass

    @abstractmethod
    def isCyclic(self) -> bool:
        pass

    @abstractmethod
    def getDataLabel(self) -> str:
        pass

    @abstractmethod
    def getDataArray(self) -> RealArrayType:
        pass

    @abstractmethod
    def __call__(self) -> RealArrayType:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._array:
            self.notifyObservers()
        elif observable is self._displayRange:
            self.notifyObservers()
        elif observable is self._transformChooser:
            self.notifyObservers()
