from __future__ import annotations
from collections.abc import Iterator
from typing import cast

import numpy

from ptychodus.api.typing import ComplexArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualize import (
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
        super().__init__('Complex', transformation, color_axis)
        self._amplitude_component = ComplexComponent.AMPLITUDE
        self._phase_component = ComplexComponent.PHASE_RAD
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

    def colorize(self, array: NumberArrayType) -> RealArrayType:
        complex_array = cast(ComplexArrayType, array)
        amplitude = self._amplitude_component.extract_component(complex_array)
        transform = self._transformation.get_strategy()
        amplitude_transformed = transform.transform(amplitude)
        phase_rad = self._phase_component.extract_component(complex_array)

        model = self._color_model.get_strategy()
        hue = (phase_rad + numpy.pi) / (2 * numpy.pi)
        return model.render_rgba(hue, self._color_normalize()(amplitude_transformed))

    def _visualize(
        self,
        array: NumberArrayType,
        pixel_geometry: PixelGeometry,
        value_min: float | None,
        value_max: float | None,
    ) -> VisualizationProduct:
        return visualize_complex_values(
            values=cast(ComplexArrayType, array),
            pixel_geometry=pixel_geometry,
            model=self._color_model.get_strategy(),
            amplitude_transform=self._transformation.get_strategy(),
            value_min=value_min,
            value_max=value_max,
            clip=False,
        )
