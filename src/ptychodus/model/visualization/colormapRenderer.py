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

from .colorAxis import ColorAxis
from .colormap import ColormapParameter
from .components import DataArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class ColormapRenderer(Renderer):
    def __init__(
        self,
        component: DataArrayComponent,
        transformation: ScalarTransformationParameter,
        colorAxis: ColorAxis,
        colormap: ColormapParameter,
    ) -> None:
        super().__init__(component.name)
        self._component = component
        self._transformation = transformation
        self._add_parameter('transformation', transformation)
        self._colorAxis = colorAxis
        self._add_group('color_axis', colorAxis, observe=True)
        self._colormap = colormap
        self._add_parameter('colormap', colormap)

    def variants(self) -> Iterator[str]:
        return self._colormap.choices()

    def getVariant(self) -> str:
        return self._colormap.get_value()

    def setVariant(self, variant: str) -> None:
        self._colormap.set_value(variant)

    def isCyclic(self) -> bool:
        return self._component.isCyclic

    def _colorize(self, valuesTransformed: RealArrayType) -> RealArrayType:
        vrange = self._colorAxis.getRange()
        norm = Normalize(vmin=vrange.lower, vmax=vrange.upper, clip=False)
        cmap = self._colormap.getPlugin()
        scalarMappable = ScalarMappable(norm, cmap)
        return scalarMappable.to_rgba(valuesTransformed)

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        values = self._component.calculate(array)
        valuesTransformed = self._transformation.transform(values)
        return self._colorize(valuesTransformed)

    def render(
        self,
        array: NumberArrayType,
        pixelGeometry: PixelGeometry,
        *,
        autoscaleColorAxis: bool,
    ) -> VisualizationProduct:
        values = self._component.calculate(array)
        valuesTransformed = self._transformation.transform(values)

        if autoscaleColorAxis:
            self._colorAxis.setToDataRange(valuesTransformed)

        rgba = self._colorize(valuesTransformed)

        return VisualizationProduct(
            value_label=self._transformation.decorateText(self._component.name),
            values=array,
            rgba=rgba,
            pixel_geometry=pixelGeometry,
        )
