from __future__ import annotations
from collections.abc import Iterator
import logging

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import NumberArrayType, VisualizationProduct

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

logger = logging.getLogger(__name__)


class VisualizationEngine(Observable, Observer):

    def __init__(self, *, isComplex: bool) -> None:
        super().__init__()
        self._rendererChooser = PluginChooser[Renderer]()
        self._transformation = ScalarTransformationParameter()
        self._colorAxis = ColorAxis()
        acyclicColormap = ColormapParameter(isCyclic=False)

        self._rendererChooser.registerPlugin(
            ColormapRenderer(RealArrayComponent(), self._transformation, self._colorAxis,
                             acyclicColormap),
            displayName='Real',
        )

        if isComplex:
            amplitudeComponent = AmplitudeArrayComponent()
            phaseComponent = PhaseInRadiansArrayComponent()
            cyclicColormap = ColormapParameter(isCyclic=True)
            colorModel = CylindricalColorModelParameter()

            self._rendererChooser.registerPlugin(
                ColormapRenderer(ImaginaryArrayComponent(), self._transformation, self._colorAxis,
                                 acyclicColormap),
                displayName='Imaginary',
            )
            self._rendererChooser.registerPlugin(
                ColormapRenderer(amplitudeComponent, self._transformation, self._colorAxis,
                                 acyclicColormap),
                displayName='Amplitude',
            )
            self._rendererChooser.registerPlugin(
                ColormapRenderer(phaseComponent, self._transformation, self._colorAxis,
                                 cyclicColormap),
                displayName='Phase',
            )
            self._rendererChooser.registerPlugin(
                ColormapRenderer(UnwrappedPhaseInRadiansArrayComponent(), self._transformation,
                                 self._colorAxis, acyclicColormap),
                displayName='Phase (Unwrapped)',
            )
            self._rendererChooser.registerPlugin(
                CylindricalColorModelRenderer(amplitudeComponent, phaseComponent,
                                              self._transformation, self._colorAxis, colorModel),
                displayName='Complex',
            )

        self._rendererChooser.addObserver(self)
        self._rendererPlugin = self._rendererChooser.currentPlugin
        self._rendererPlugin.strategy.addObserver(self)

    def renderers(self) -> Iterator[str]:
        for plugin in self._rendererChooser:
            yield plugin.displayName

    def getRenderer(self) -> str:
        return self._rendererPlugin.displayName

    def setRenderer(self, value: str) -> None:
        self._rendererChooser.setCurrentPluginByName(value)

    def isRendererCyclic(self) -> bool:
        return self._rendererPlugin.strategy.isCyclic()

    def transformations(self) -> Iterator[str]:
        return self._transformation.choices()

    def getTransformation(self) -> str:
        return self._transformation.getValue()

    def setTransformation(self, value: str) -> None:
        self._transformation.setValue(value)

    def variants(self) -> Iterator[str]:
        return self._rendererPlugin.strategy.variants()

    def getVariant(self) -> str:
        return self._rendererPlugin.strategy.getVariant()

    def setVariant(self, value: str) -> None:
        return self._rendererPlugin.strategy.setVariant(value)

    def getMinDisplayValue(self) -> float:
        return self._colorAxis.lower.getValue()

    def setMinDisplayValue(self, value: float) -> None:
        self._colorAxis.lower.setValue(value)

    def getMaxDisplayValue(self) -> float:
        return self._colorAxis.upper.getValue()

    def setMaxDisplayValue(self, value: float) -> None:
        self._colorAxis.upper.setValue(value)

    def setDisplayValueRange(self, lower: float, upper: float) -> None:
        self._colorAxis.setRange(lower, upper)

    def render(self, array: NumberArrayType, pixelGeometry: PixelGeometry) -> VisualizationProduct:
        return self._rendererPlugin.strategy.render(array, pixelGeometry)

    def update(self, observable: Observable) -> None:
        if observable is self._rendererChooser:
            self._rendererPlugin.strategy.removeObserver(self)
            self._rendererPlugin = self._rendererChooser.currentPlugin
            self._rendererPlugin.strategy.addObserver(self)
            self.notifyObservers()
        elif observable is self._rendererPlugin.strategy:
            self.notifyObservers()
