from __future__ import annotations
from collections.abc import Iterator

from matplotlib.colors import Normalize
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import (
    ComplexComponent,
    NumberArrayType,
    RealArrayType,
    VisualizationProduct,
    visualize_complex_values,
)

from .color_axis import ColorAxis
from .color_model import CylindricalColorModelParameter
from .renderer import Renderer
from .transformation import ScalarTransformationParameter


class CylindricalColorModelRenderer(Renderer):
    def __init__(
        self,
        transformation: ScalarTransformationParameter,
        color_axis: ColorAxis,
    ) -> None:
        super().__init__('Complex')
        self._amplitude_component = ComplexComponent.AMPLITUDE
        self._phase_component = ComplexComponent.PHASE_RAD
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
        norm = Normalize(
            vmin=self._color_axis.lower.get_value(),
            vmax=self._color_axis.upper.get_value(),
            clip=False,
        )

        model = self._color_model.get_strategy()
        h = (phase_rad + numpy.pi) / (2 * numpy.pi)
        return model.render_rgba(h, norm(amplitude))

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        amplitude = self._amplitude_component.extract_component(array)
        transform = self._transformation.get_strategy()
        amplitude_transformed = transform.transform(amplitude)
        phase_rad = self._phase_component.extract_component(array)
        return self._colorize(amplitude_transformed, phase_rad)

    def render(
        self, array: NumberArrayType, pixel_geometry: PixelGeometry, *, autoscale_color_axis: bool
    ) -> VisualizationProduct:
        value_min: float | None = None
        value_max: float | None = None

        if not autoscale_color_axis:
            value_min = self._color_axis.lower.get_value()
            value_max = self._color_axis.upper.get_value()

        product = visualize_complex_values(
            values=array,
            pixel_geometry=pixel_geometry,
            model=self._color_model.get_strategy(),
            amplitude_transform=self._transformation.get_strategy(),
            value_min=value_min,
            value_max=value_max,
            clip=False,
        )

        if autoscale_color_axis:
            color_value_range = product.get_color_value_range()
            self._color_axis.set_to_data_range(color_value_range.lower, color_value_range.upper)

        return product
