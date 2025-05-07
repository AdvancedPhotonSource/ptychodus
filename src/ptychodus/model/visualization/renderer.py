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

    def get_name(self) -> str:
        return self._name.get_value()

    @abstractmethod
    def variants(self) -> Iterator[str]:
        pass

    @abstractmethod
    def get_variant(self) -> str:
        pass

    @abstractmethod
    def set_variant(self, variant: str) -> None:
        pass

    @abstractmethod
    def is_cyclic(self) -> bool:
        pass

    @abstractmethod
    def colorize(self, array: NumberArrayType) -> RealArrayType:
        pass

    @abstractmethod
    def render(
        self, array: NumberArrayType, pixel_geometry: PixelGeometry, *, autoscale_color_axis: bool
    ) -> VisualizationProduct:
        pass
