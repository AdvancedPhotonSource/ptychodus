from abc import abstractmethod
from collections.abc import Iterator

from matplotlib.colors import Normalize

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualize import (
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
)
from ptychodus.model.parameters import ParameterGroup

from .color_axis import ColorAxis
from .transformation import ScalarTransformationParameter


class Renderer(ParameterGroup):
    """Base for the strategies a VisualizationEngine chooses between.

    Owns the two pieces every renderer shares -- the scalar transformation applied
    before coloring, and the color axis the result is normalized against -- so a
    subclass supplies only the visualize call that distinguishes it.
    """

    def __init__(
        self,
        name: str,
        transformation: ScalarTransformationParameter,
        color_axis: ColorAxis,
    ) -> None:
        super().__init__()
        self._name = self.create_string_parameter('name', name)
        self._transformation = transformation
        self._add_parameter('transformation', transformation)
        self._color_axis = color_axis
        self._add_group('color_axis', color_axis, observe=True)

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

    def _color_normalize(self) -> Normalize:
        """Normalization over the current color axis, for the legend-ramp path."""
        return Normalize(
            vmin=self._color_axis.lower.get_value(),
            vmax=self._color_axis.upper.get_value(),
            clip=False,
        )

    @abstractmethod
    def colorize(self, array: NumberArrayType) -> RealArrayType:
        pass

    @abstractmethod
    def _visualize(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        value_min: float | None,
        value_max: float | None,
    ) -> VisualizationProduct:
        """Build the product over the given color bounds; None means autoscale."""

    def render(
        self, array: NumberArrayType, pixel_geometry: PixelGeometry, *, autoscale_color_axis: bool
    ) -> VisualizationProduct:
        value_min: float | None = None
        value_max: float | None = None

        if not autoscale_color_axis:
            value_min = self._color_axis.lower.get_value()
            value_max = self._color_axis.upper.get_value()

        product = self._visualize(array, pixel_geometry, value_min, value_max)

        # Writing the realized range back is what keeps the data-range slider and the
        # engine agreeing after an autoscaled render.
        if autoscale_color_axis:
            color_value_range = product.get_color_value_range()
            self._color_axis.set_to_data_range(color_value_range.lower, color_value_range.upper)

        return product
