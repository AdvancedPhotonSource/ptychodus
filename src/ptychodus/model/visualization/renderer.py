from abc import abstractmethod
from collections.abc import Iterator

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.visualization import (
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
)


class Renderer(ParameterGroup):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = self.create_string_parameter('name', name)

    def getName(self) -> str:
        return self._name.get_value()

    @abstractmethod
    def variants(self) -> Iterator[str]:
        pass

    @abstractmethod
    def getVariant(self) -> str:
        pass

    @abstractmethod
    def setVariant(self, variant: str) -> None:
        pass

    @abstractmethod
    def isCyclic(self) -> bool:
        pass

    @abstractmethod
    def colorize(self, array: NumberArrayType) -> RealArrayType:
        pass

    @abstractmethod
    def render(
        self, array: NumberArrayType, pixelGeometry: PixelGeometry, *, autoscaleColorAxis: bool
    ) -> VisualizationProduct:
        pass
