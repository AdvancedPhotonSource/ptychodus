from __future__ import annotations
from collections.abc import Iterator
from matplotlib.cm import ScalarMappable

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualize import (
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
        super().__init__(component.name, transformation, color_axis)
        self._component = component
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

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        values = self._component.extract_component(array)
        transform = self._transformation.get_strategy()
        values_transformed = transform.transform(values)
        scalar_mappable = ScalarMappable(self._color_normalize(), self._colormap.get_strategy())
        return scalar_mappable.to_rgba(values_transformed)

    def _visualize(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        value_min: float | None,
        value_max: float | None,
    ) -> VisualizationProduct:
        return visualize_complex_component(
            values=array,
            pixel_geometry=pixel_geometry,
            component=self._component,
            colormap=self._colormap.get_strategy(),
            transform=self._transformation.get_strategy(),
            value_min=value_min,
            value_max=value_max,
            clip=False,
        )
