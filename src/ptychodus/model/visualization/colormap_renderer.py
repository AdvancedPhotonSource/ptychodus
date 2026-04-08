from __future__ import annotations
from collections.abc import Iterator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import (
    ComplexComponent,
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
    visualize_complex_component,
)

from .color_axis import ColorAxis
from .colormap import ColormapParameter
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class ColormapRenderer(Renderer):
    def __init__(
        self,
        component: ComplexComponent,
        transformation: ScalarTransformationParameter,
        color_axis: ColorAxis,
        colormap: ColormapParameter,
    ) -> None:
        super().__init__(component.name)
        self._component = component
        self._transformation = transformation
        self._add_parameter('transformation', transformation)
        self._color_axis = color_axis
        self._add_group('color_axis', color_axis, observe=True)
        self._colormap = colormap
        self._add_parameter('colormap', colormap)

    def variants(self) -> Iterator[str]:
        return self._colormap.choices()

    def get_variant(self) -> str:
        return self._colormap.get_value()

    def set_variant(self, variant: str) -> None:
        self._colormap.set_value(variant)

    def is_cyclic(self) -> bool:
        return self._component.is_cyclic

    def _colorize(self, values_transformed: RealArrayType) -> RealArrayType:
        norm = Normalize(
            vmin=self._color_axis.lower.get_value(),
            vmax=self._color_axis.upper.get_value(),
            clip=False,
        )
        cmap = self._colormap.get_strategy()
        scalar_mappable = ScalarMappable(norm, cmap)
        return scalar_mappable.to_rgba(values_transformed)

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        values = self._component.extract_component(array)
        transform = self._transformation.get_strategy()
        values_transformed = transform.transform(values)
        return self._colorize(values_transformed)

    def render(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        *,
        autoscale_color_axis: bool,
    ) -> VisualizationProduct:
        value_min: float | None = None
        value_max: float | None = None

        if not autoscale_color_axis:
            value_min = self._color_axis.lower.get_value()
            value_max = self._color_axis.upper.get_value()

        product = visualize_complex_component(
            values=array,
            pixel_geometry=pixel_geometry,
            component=self._component,
            colormap=self._colormap.get_strategy(),
            transform=self._transformation.get_strategy(),
            value_min=value_min,
            value_max=value_max,
            clip=False,
        )

        if autoscale_color_axis:
            color_value_range = product.get_color_value_range()
            self._color_axis.set_to_data_range(color_value_range.lower, color_value_range.upper)

        return product
