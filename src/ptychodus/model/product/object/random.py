from __future__ import annotations
from collections.abc import Sequence

import numpy
from scipy.ndimage import gaussian_filter

from ptychodus.api.common import TWO_PI_J
from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder
from .settings import ObjectSettings


class RandomObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'random')
        self._rng = rng
        self._settings = settings

        self.amplitude_mean = settings.amplitude_mean.copy()
        self._add_parameter('amplitude_mean', self.amplitude_mean)
        self.amplitude_deviation = settings.amplitude_deviation.copy()
        self._add_parameter('amplitude_deviation', self.amplitude_deviation)
        self.phase_deviation_turns = settings.phase_deviation_turns.copy()
        self._add_parameter('phase_deviation_turns', self.phase_deviation_turns)
        self.blur_deviation_px = settings.blur_deviation_px.copy()
        self._add_parameter('blur_deviation_px', self.blur_deviation_px)

    def copy(self) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        geometry = geometry_provider.get_object_geometry()
        object_shape = (1, geometry.height_px, geometry.width_px)

        amplitude = self._rng.normal(
            self.amplitude_mean.get_value(),
            self.amplitude_deviation.get_value(),
            object_shape,
        )
        phase = self._rng.normal(
            0.0,
            self.phase_deviation_turns.get_value(),
            object_shape,
        )

        blur_sigma = self.blur_deviation_px.get_value()

        if blur_sigma > 0.0:
            # TODO account for pixel aspect ratio in blur calculation
            amplitude = gaussian_filter(amplitude, sigma=blur_sigma)
            phase = gaussian_filter(phase, sigma=blur_sigma)

        array = amplitude * numpy.exp(TWO_PI_J * phase)

        return self._create_object(
            array=array.astype('complex'),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
