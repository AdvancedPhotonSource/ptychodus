from abc import ABC, abstractmethod
from decimal import Decimal

from ...api.geometry import Interval
from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from .displayRange import DisplayRange
from .visarray import VisualizationArrayComponent


class Colorizer(Observable, Observer, ABC):

    def __init__(self, arrayComponent: VisualizationArrayComponent,
                 transformChooser: PluginChooser[ScalarTransformation],
                 displayRange: DisplayRange) -> None:
        super().__init__()
        self._arrayComponent = arrayComponent
        self._arrayComponent.addObserver(self)
        self._transformChooser = transformChooser
        self._transformChooser.addObserver(self)
        self._displayRange = displayRange
        self._displayRange.addObserver(self)

    @property
    def name(self) -> str:
        return self._arrayComponent.name

    def getScalarTransformationList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    @abstractmethod
    def getVariantList(self) -> list[str]:
        pass

    @abstractmethod
    def getVariant(self) -> str:
        pass

    @abstractmethod
    def setVariant(self, name: str) -> None:
        pass

    @abstractmethod
    def getDataRange(self) -> Interval[Decimal]:
        pass

    @abstractmethod
    def __call__(self) -> RealArrayType:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._arrayComponent:
            self.notifyObservers()
        elif observable is self._transformChooser:
            self.notifyObservers()
        elif observable is self._displayRange:
            self.notifyObservers()
