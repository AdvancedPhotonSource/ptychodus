from abc import abstractmethod
from collections.abc import Iterator

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.visualization import NumberArrayType, VisualizationProduct


class Renderer(ParameterRepository):

    def __init__(self, name: str) -> None:
        super().__init__('renderer')
        self._name = self._registerStringParameter('name', name)

    def getName(self) -> str:
        return self._name.getValue()

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
    def render(self, array: NumberArrayType, pixelGeometry: PixelGeometry, *,
               autoscaleColorAxis: bool) -> VisualizationProduct:
        pass
