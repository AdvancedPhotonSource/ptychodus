from __future__ import annotations
from collections.abc import Iterator
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import (
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
)

from .color_axis import ColorAxis
from .colormap import ColormapParameter
from .components import DataArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class ColormapRenderer(Renderer):
    def __init__(
        self,
        component: DataArrayComponent,
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
        vrange = self._color_axis.get_range()
        norm = Normalize(vmin=vrange.lower, vmax=vrange.upper, clip=False)
        cmap = self._colormap.get_plugin()
        scalar_mappable = ScalarMappable(norm, cmap)
        return scalar_mappable.to_rgba(values_transformed)

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        values = self._component.calculate(array)
        values_transformed = self._transformation.transform(values)
        return self._colorize(values_transformed)

    def render(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        *,
        autoscale_color_axis: bool,
    ) -> VisualizationProduct:
        values = self._component.calculate(array)
        values_transformed = self._transformation.transform(values)

        if autoscale_color_axis:
            self._color_axis.set_to_data_range(values_transformed)

        rgba = self._colorize(values_transformed)

        return VisualizationProduct(
            value_label=self._transformation.decorate_text(self._component.name),
            values=array,
            rgba=rgba,
            pixel_geometry=pixel_geometry,
        )
