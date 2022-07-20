from abc import ABC, abstractmethod

from ...api.image import RealArrayType
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from .visarray import VisualizationArrayComponent


class Colorizer(Observable, Observer):

    def __init__(self, name: str,
                 componentChooser: PluginChooser[VisualizationArrayComponent]) -> None:
        super().__init__()
        self._name = name
        self._componentChooser = componentChooser
        self._componentChooser.addObserver(self)
        self._component.addObserver(self)

    @property
    def name(self) -> str:
        return self._name

    def getArrayComponentList(self) -> list[str]:
        return self._componentChooser.getDisplayNameList()

    def getArrayComponent(self) -> str:
        return self._componentChooser.getCurrentDisplayName()

    @property
    def _component(self) -> VisualizationArrayComponent:
        return self._componentChooser.getCurrentStrategy()

    def setArrayComponent(self, name: str) -> None:
        self._component.removeObserver(self)
        self._componentChooser.setFromDisplayName(name)
        self._component.addObserver(self)

    def getScalarTransformationList(self) -> list[str]:
        return self._component.getScalarTransformationList()

    def getScalarTransformation(self) -> str:
        return self._component.getScalarTransformation()

    def setScalarTransformation(self, name: str) -> None:
        self._component.setScalarTransformation(name)

    @abstractmethod
    def __call__(self) -> RealArrayType:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._component:
            self.notifyObservers()
        elif observable is self._componentChooser:
            self.notifyObservers()
