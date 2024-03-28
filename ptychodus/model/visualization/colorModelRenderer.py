from __future__ import annotations
from collections.abc import Iterator
from typing import override

from matplotlib.colors import Normalize
import numpy

from ptychodus.api.patterns import PixelGeometry
from ptychodus.api.visualization import NumberArrayType, VisualizationProduct

from .colorAxis import ColorAxis
from .colorModel import CylindricalColorModelParameter
from .components import AmplitudeArrayComponent, PhaseInRadiansArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class CylindricalColorModelRenderer(Renderer):

    def __init__(self, amplitudeComponent: AmplitudeArrayComponent,
                 phaseComponent: PhaseInRadiansArrayComponent,
                 transformation: ScalarTransformationParameter, colorAxis: ColorAxis,
                 colorModel: CylindricalColorModelParameter) -> None:
        super().__init__('Complex')
        self._amplitudeComponent = amplitudeComponent
        self._phaseComponent = phaseComponent
        self._transformation = transformation
        self._registerParameter('transformation', transformation)
        self._colorAxis = colorAxis
        self._addParameterRepository(colorAxis, observe=True)
        self._colorModel = colorModel
        self._registerParameter('color_model', colorModel)

    @override
    def variants(self) -> Iterator[str]:
        return self._colorModel.choices()

    @override
    def getVariant(self) -> str:
        return self._colorModel.getValue()

    @override
    def setVariant(self, variant: str) -> None:
        self._colorModel.setValue(variant)

    @override
    def isCyclic(self) -> bool:
        return True

    @override
    def render(self, array: NumberArrayType, pixelGeometry: PixelGeometry, *,
               autoscaleColorAxis: bool) -> VisualizationProduct:
        amplitude = self._amplitudeComponent.calculate(array)
        phaseInRadians = self._phaseComponent.calculate(array)

        transform = self._transformation.getPlugin()
        amplitudeTransformed = transform(amplitude)

        if autoscaleColorAxis:
            self._colorAxis.setToDataRange(amplitudeTransformed)

        norm = Normalize(vmin=self._colorAxis.lower.getValue(),
                         vmax=self._colorAxis.upper.getValue(),
                         clip=False)

        model = numpy.vectorize(self._colorModel.getPlugin())
        h = (phaseInRadians + numpy.pi) / (2 * numpy.pi)
        r, g, b, a = model(h, norm(amplitudeTransformed))
        rgba = numpy.stack((r, g, b, a), axis=-1)

        return VisualizationProduct(
            valueLabel=transform.decorateText(self._amplitudeComponent.name),
            values=array,
            rgba=rgba,
            pixelGeometry=pixelGeometry,
        )
