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

from .color_axis import ColorAxis
from .color_model_renderer import CylindricalColorModelRenderer
from .colormap import ColormapParameter
from .colormap_renderer import ColormapRenderer
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
    def __init__(self, *, is_complex: bool) -> None:
        super().__init__()
        self._renderer_chooser = PluginChooser[Renderer]()
        self._transformation = ScalarTransformationParameter()
        self._color_axis = ColorAxis()
        acyclic_colormap = ColormapParameter(is_cyclic=False)

        self._renderer_chooser.register_plugin(
            ColormapRenderer(
                RealArrayComponent(),
                self._transformation,
                self._color_axis,
                acyclic_colormap,
            ),
            display_name='Real',
        )

        if is_complex:
            amplitude_component = AmplitudeArrayComponent()
            phase_component = PhaseInRadiansArrayComponent()
            cyclic_colormap = ColormapParameter(is_cyclic=True)

            self._renderer_chooser.register_plugin(
                ColormapRenderer(
                    ImaginaryArrayComponent(),
                    self._transformation,
                    self._color_axis,
                    acyclic_colormap,
                ),
                display_name='Imaginary',
            )
            self._renderer_chooser.register_plugin(
                ColormapRenderer(
                    amplitude_component,
                    self._transformation,
                    self._color_axis,
                    acyclic_colormap,
                ),
                display_name='Amplitude',
            )
            self._renderer_chooser.register_plugin(
                ColormapRenderer(
                    phase_component,
                    self._transformation,
                    self._color_axis,
                    cyclic_colormap,
                ),
                display_name='Phase',
            )
            self._renderer_chooser.register_plugin(
                ColormapRenderer(
                    UnwrappedPhaseInRadiansArrayComponent(),
                    self._transformation,
                    self._color_axis,
                    acyclic_colormap,
                ),
                display_name='Phase (Unwrapped)',
            )
            self._renderer_chooser.register_plugin(
                CylindricalColorModelRenderer(
                    amplitude_component,
                    phase_component,
                    self._transformation,
                    self._color_axis,
                ),
                display_name='Complex',
            )
            self._renderer_chooser.set_current_plugin('Complex')

        self._renderer_chooser.add_observer(self)
        self._renderer_plugin = self._renderer_chooser.get_current_plugin()
        self._renderer_plugin.strategy.add_observer(self)

    def renderers(self) -> Iterator[str]:
        for plugin in self._renderer_chooser:
            yield plugin.display_name

    def get_renderer(self) -> str:
        return self._renderer_plugin.display_name

    def set_renderer(self, value: str) -> None:
        self._renderer_chooser.set_current_plugin(value)

    def is_renderer_cyclic(self) -> bool:
        return self._renderer_plugin.strategy.is_cyclic()

    def transformations(self) -> Iterator[str]:
        return self._transformation.choices()

    def get_transformation(self) -> str:
        return self._transformation.get_value()

    def set_transformation(self, value: str) -> None:
        self._transformation.set_value(value)

    def variants(self) -> Iterator[str]:
        return self._renderer_plugin.strategy.variants()

    def get_variant(self) -> str:
        return self._renderer_plugin.strategy.get_variant()

    def set_variant(self, value: str) -> None:
        return self._renderer_plugin.strategy.set_variant(value)

    def get_min_display_value(self) -> float:
        return self._color_axis.lower.get_value()

    def set_min_display_value(self, value: float) -> None:
        self._color_axis.lower.set_value(value)

    def get_max_display_value(self) -> float:
        return self._color_axis.upper.get_value()

    def set_max_display_value(self, value: float) -> None:
        self._color_axis.upper.set_value(value)

    def set_display_value_range(self, lower: float, upper: float) -> None:
        self._color_axis.set_range(lower, upper)

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        return self._renderer_plugin.strategy.colorize(array)

    def render(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        *,
        autoscale_color_axis: bool,
    ) -> VisualizationProduct:
        return self._renderer_plugin.strategy.render(
            array, pixel_geometry, autoscale_color_axis=autoscale_color_axis
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._renderer_chooser:
            self._renderer_plugin.strategy.remove_observer(self)
            self._renderer_plugin = self._renderer_chooser.get_current_plugin()
            self._renderer_plugin.strategy.add_observer(self)
            self.notify_observers()
        elif observable is self._renderer_plugin.strategy:
            self.notify_observers()
