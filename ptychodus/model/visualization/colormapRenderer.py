from __future__ import annotations
from collections.abc import Iterator
from typing import override

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import NumberArrayType, VisualizationProduct

from .colorAxis import ColorAxis
from .colormap import ColormapParameter
from .components import DataArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class ColormapRenderer(Renderer):

    def __init__(self, component: DataArrayComponent,
                 transformation: ScalarTransformationParameter, colorAxis: ColorAxis,
                 colormap: ColormapParameter) -> None:
        super().__init__(component.name)
        self._component = component
        self._transformation = transformation
        self._registerParameter('transformation', transformation)
        self._colorAxis = colorAxis
        self._addParameterRepository(colorAxis, observe=True)
        self._colormap = colormap
        self._registerParameter('colormap', colormap)

    @override
    def variants(self) -> Iterator[str]:
        return self._colormap.choices()

    @override
    def getVariant(self) -> str:
        return self._colormap.getValue()

    @override
    def setVariant(self, variant: str) -> None:
        self._colormap.setValue(variant)

    @override
    def isCyclic(self) -> bool:
        return self._component.isCyclic

    @override
    def render(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> VisualizationProduct:
        values = self._component.calculate(array)

        transform = self._transformation.getPlugin()
        valuesTransformed = transform(values)

        if False:  # FIXME autoscaleColorAxis:
            self._colorAxis.setToDataRange(valuesTransformed)

        norm = Normalize(vmin=self._colorAxis.lower.getValue(),
                         vmax=self._colorAxis.upper.getValue(),
                         clip=False)
        cmap = self._colormap.getPlugin()
        scalarMappable = ScalarMappable(norm, cmap)
        rgba = scalarMappable.to_rgba(valuesTransformed)

        return VisualizationProduct(
            valueLabel=transform.decorateText(self._component.name),
            values=array,
            rgba=rgba,
            pixelGeometry=pixelGeometry,
        )
