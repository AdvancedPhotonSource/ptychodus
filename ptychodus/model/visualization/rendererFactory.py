from collections.abc import Sequence

from .colorAxis import ColorAxis
from .colorModel import CylindricalColorModelParameter
from .colorModelRenderer import CylindricalColorModelRenderer
from .colormap import ColormapParameter
from .colormapRenderer import ColormapRenderer
from .components import (AmplitudeArrayComponent, ImaginaryArrayComponent,
                         PhaseInRadiansArrayComponent, RealArrayComponent,
                         UnwrappedPhaseInRadiansArrayComponent)

from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class RendererFactory:

    def createRenderers(self, *, isComplex: bool) -> Sequence[Renderer]:
        transformation = ScalarTransformationParameter()
        colorAxis = ColorAxis()
        acyclicColormap = ColormapParameter(isCyclic=False)
        cyclicColormap = ColormapParameter(isCyclic=True)

        renderers: list[Renderer] = [
            ColormapRenderer(RealArrayComponent(), transformation, colorAxis, acyclicColormap),
        ]

        if isComplex:
            amplitudeComponent = AmplitudeArrayComponent()
            phaseComponent = PhaseInRadiansArrayComponent()
            colorModel = CylindricalColorModelParameter()

            renderers.extend([
                ColormapRenderer(ImaginaryArrayComponent(), transformation, colorAxis,
                                 acyclicColormap),
                ColormapRenderer(amplitudeComponent, transformation, colorAxis, acyclicColormap),
                ColormapRenderer(phaseComponent, transformation, colorAxis, cyclicColormap),
                ColormapRenderer(UnwrappedPhaseInRadiansArrayComponent(), transformation,
                                 colorAxis, acyclicColormap),
                CylindricalColorModelRenderer(amplitudeComponent, phaseComponent, transformation,
                                              colorAxis, colorModel),
            ])

        return renderers
