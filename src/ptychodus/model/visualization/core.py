from __future__ import annotations
from collections.abc import Iterator
import logging

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import (
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
)

from .colorAxis import ColorAxis
from .colorModelRenderer import CylindricalColorModelRenderer
from .colormap import ColormapParameter
from .colormapRenderer import ColormapRenderer
from .components import (
    AmplitudeArrayComponent,
    ImaginaryArrayComponent,
    PhaseInRadiansArrayComponent,
    RealArrayComponent,
    UnwrappedPhaseInRadiansArrayComponent,
)

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

        self._rendererChooser.register_plugin(
            ColormapRenderer(
                RealArrayComponent(),
                self._transformation,
                self._colorAxis,
                acyclicColormap,
            ),
            display_name='Real',
        )

        if isComplex:
            amplitudeComponent = AmplitudeArrayComponent()
            phaseComponent = PhaseInRadiansArrayComponent()
            cyclicColormap = ColormapParameter(isCyclic=True)

            self._rendererChooser.register_plugin(
                ColormapRenderer(
                    ImaginaryArrayComponent(),
                    self._transformation,
                    self._colorAxis,
                    acyclicColormap,
                ),
                display_name='Imaginary',
            )
            self._rendererChooser.register_plugin(
                ColormapRenderer(
                    amplitudeComponent,
                    self._transformation,
                    self._colorAxis,
                    acyclicColormap,
                ),
                display_name='Amplitude',
            )
            self._rendererChooser.register_plugin(
                ColormapRenderer(
                    phaseComponent,
                    self._transformation,
                    self._colorAxis,
                    cyclicColormap,
                ),
                display_name='Phase',
            )
            self._rendererChooser.register_plugin(
                ColormapRenderer(
                    UnwrappedPhaseInRadiansArrayComponent(),
                    self._transformation,
                    self._colorAxis,
                    acyclicColormap,
                ),
                display_name='Phase (Unwrapped)',
            )
            self._rendererChooser.register_plugin(
                CylindricalColorModelRenderer(
                    amplitudeComponent,
                    phaseComponent,
                    self._transformation,
                    self._colorAxis,
                ),
                display_name='Complex',
            )
            self._rendererChooser.set_current_plugin('Complex')

        self._rendererChooser.addObserver(self)
        self._rendererPlugin = self._rendererChooser.get_current_plugin()
        self._rendererPlugin.strategy.addObserver(self)

    def renderers(self) -> Iterator[str]:
        for plugin in self._rendererChooser:
            yield plugin.display_name

    def getRenderer(self) -> str:
        return self._rendererPlugin.display_name

    def setRenderer(self, value: str) -> None:
        self._rendererChooser.set_current_plugin(value)

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

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        return self._rendererPlugin.strategy.colorize(array)

    def render(
        self,
        array: NumberArrayType,
        pixelGeometry: PixelGeometry,
        *,
        autoscaleColorAxis: bool,
    ) -> VisualizationProduct:
        return self._rendererPlugin.strategy.render(
            array, pixelGeometry, autoscaleColorAxis=autoscaleColorAxis
        )

    def update(self, observable: Observable) -> None:
        if observable is self._rendererChooser:
            self._rendererPlugin.strategy.removeObserver(self)
            self._rendererPlugin = self._rendererChooser.get_current_plugin()
            self._rendererPlugin.strategy.addObserver(self)
            self.notifyObservers()
        elif observable is self._rendererPlugin.strategy:
            self.notifyObservers()
