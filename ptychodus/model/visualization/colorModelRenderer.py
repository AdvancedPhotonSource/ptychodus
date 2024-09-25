from __future__ import annotations
from collections.abc import Iterator
from matplotlib.colors import Normalize
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import (
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
)

from .colorAxis import ColorAxis
from .colorModel import CylindricalColorModelParameter
from .components import AmplitudeArrayComponent, PhaseInRadiansArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class CylindricalColorModelRenderer(Renderer):

    def __init__(
        self,
        amplitudeComponent: AmplitudeArrayComponent,
        phaseComponent: PhaseInRadiansArrayComponent,
        transformation: ScalarTransformationParameter,
        colorAxis: ColorAxis,
    ) -> None:
        super().__init__("Complex")
        self._amplitudeComponent = amplitudeComponent
        self._phaseComponent = phaseComponent
        self._transformation = transformation
        self._addParameter("transformation", transformation)
        self._colorAxis = colorAxis
        self._addGroup("color_axis", colorAxis, observe=True)
        self._colorModel = CylindricalColorModelParameter(self)
        self._addParameter("color_model", self._colorModel)

    def variants(self) -> Iterator[str]:
        return self._colorModel.choices()

    def getVariant(self) -> str:
        return self._colorModel.getValue()

    def setVariant(self, variant: str) -> None:
        self._colorModel.setValue(variant)

    def isCyclic(self) -> bool:
        return True

    def _colorize(self, amplitudeTransformed: RealArrayType,
                  phaseInRadians: RealArrayType) -> RealArrayType:
        vrange = self._colorAxis.getRange()
        norm = Normalize(vmin=vrange.lower, vmax=vrange.upper, clip=False)

        model = numpy.vectorize(self._colorModel.getPlugin())
        h = (phaseInRadians + numpy.pi) / (2 * numpy.pi)
        r, g, b, a = model(h, norm(amplitudeTransformed))
        return numpy.stack((r, g, b, a), axis=-1)

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        amplitude = self._amplitudeComponent.calculate(array)
        amplitudeTransformed = self._transformation.transform(amplitude)
        phaseInRadians = self._phaseComponent.calculate(array)
        return self._colorize(amplitudeTransformed, phaseInRadians)

    def render(
        self,
        array: NumberArrayType,
        pixelGeometry: PixelGeometry,
        *,
        autoscaleColorAxis: bool,
    ) -> VisualizationProduct:
        amplitude = self._amplitudeComponent.calculate(array)
        amplitudeTransformed = self._transformation.transform(amplitude)
        phaseInRadians = self._phaseComponent.calculate(array)

        if autoscaleColorAxis:
            self._colorAxis.setToDataRange(amplitudeTransformed)

        rgba = self._colorize(amplitudeTransformed, phaseInRadians)

        return VisualizationProduct(
            valueLabel=self._transformation.decorateText(self._amplitudeComponent.name),
            values=array,
            rgba=rgba,
            pixelGeometry=pixelGeometry,
        )
