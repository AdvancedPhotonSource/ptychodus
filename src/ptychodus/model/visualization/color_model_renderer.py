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

from .color_axis import ColorAxis
from .color_model import CylindricalColorModelParameter
from .components import AmplitudeArrayComponent, PhaseInRadiansArrayComponent
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class CylindricalColorModelRenderer(Renderer):
    def __init__(
        self,
        amplitude_component: AmplitudeArrayComponent,
        phase_component: PhaseInRadiansArrayComponent,
        transformation: ScalarTransformationParameter,
        color_axis: ColorAxis,
    ) -> None:
        super().__init__('Complex')
        self._amplitude_component = amplitude_component
        self._phase_component = phase_component
        self._transformation = transformation
        self._add_parameter('transformation', transformation)
        self._color_axis = color_axis
        self._add_group('color_axis', color_axis, observe=True)
        self._color_model = CylindricalColorModelParameter()
        self._add_parameter('color_model', self._color_model)

    def variants(self) -> Iterator[str]:
        return self._color_model.choices()

    def get_variant(self) -> str:
        return self._color_model.get_value()

    def set_variant(self, variant: str) -> None:
        self._color_model.set_value(variant)

    def is_cyclic(self) -> bool:
        return True

    def _colorize(self, amplitude: RealArrayType, phase_rad: RealArrayType) -> RealArrayType:
        vrange = self._color_axis.get_range()
        norm = Normalize(vmin=vrange.lower, vmax=vrange.upper, clip=False)

        model = self._color_model.get_plugin()
        h = (phase_rad + numpy.pi) / (2 * numpy.pi)
        return model(h, norm(amplitude))

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        amplitude = self._amplitude_component.calculate(array)
        amplitude_transformed = self._transformation.transform(amplitude)
        phase_rad = self._phase_component.calculate(array)
        return self._colorize(amplitude_transformed, phase_rad)

    def render(
        self, array: NumberArrayType, pixel_geometry: PixelGeometry, *, autoscale_color_axis: bool
    ) -> VisualizationProduct:
        amplitude = self._amplitude_component.calculate(array)
        amplitude_transformed = self._transformation.transform(amplitude)
        phase_rad = self._phase_component.calculate(array)

        if autoscale_color_axis:
            self._color_axis.set_to_data_range(amplitude_transformed)

        rgba = self._colorize(amplitude_transformed, phase_rad)

        return VisualizationProduct(
            value_label=self._transformation.decorate_text(self._amplitude_component.name),
            values=array,
            rgba=rgba,
            pixel_geometry=pixel_geometry,
        )
