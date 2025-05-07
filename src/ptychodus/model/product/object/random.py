from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.model.phase_unwrapper import PhaseUnwrapper

from .builder import ObjectBuilder
from .settings import ObjectSettings


class RandomObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'random')
        self._rng = rng
        self._settings = settings

        self.extra_padding_x = settings.extra_padding_x.copy()
        self._add_parameter('extra_padding_x', self.extra_padding_x)
        self.extra_padding_y = settings.extra_padding_y.copy()
        self._add_parameter('extra_padding_y', self.extra_padding_y)

        self.amplitude_mean = settings.amplitude_mean.copy()
        self._add_parameter('amplitude_mean', self.amplitude_mean)
        self.amplitude_deviation = settings.amplitude_deviation.copy()
        self._add_parameter('amplitude_deviation', self.amplitude_deviation)

        self.phase_deviation = settings.phase_deviation.copy()
        self._add_parameter('phase_deviation', self.phase_deviation)

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
        height_px = geometry.height_px + 2 * self.extra_padding_y.get_value()
        width_px = geometry.width_px + 2 * self.extra_padding_x.get_value()
        object_shape = (1 + len(layer_spacing_m), height_px, width_px)

        amplitude = self._rng.normal(
            self.amplitude_mean.get_value(),
            self.amplitude_deviation.get_value(),
            object_shape,
        )
        phase = self._rng.normal(
            0.0,
            self.phase_deviation.get_value(),
            object_shape,
        )

        return Object(
            array=numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )


class UserObjectBuilder(ObjectBuilder):  # TODO use
    def __init__(self, object_: Object, settings: ObjectSettings) -> None:
        """Create an object from an existing object with a potentially
        different number of slices.

        If the new object is supposed to be a multislice object with a
        different number of slices than the existing object, the object is
        created as
        `abs(o) ** (1 / nSlices) * exp(i * unwrapPhase(o) / nSlices)`.
        Otherwise, the object is copied as is.
        """
        super().__init__(settings, 'user')
        self._existing_object = object_
        self._settings = settings

    def copy(self) -> UserObjectBuilder:
        builder = UserObjectBuilder(self._existing_object, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        geometry = self._existing_object.get_geometry()
        existing_object_array = self._existing_object.get_array()
        num_slices = len(layer_spacing_m) + 1

        if num_slices > 1 and num_slices != existing_object_array.shape[0]:
            amplitude = numpy.abs(existing_object_array[0:1]) ** (1.0 / num_slices)
            amplitude = amplitude.repeat(num_slices, axis=0)
            phase = PhaseUnwrapper().unwrap(existing_object_array[0])[None, ...] / num_slices
            phase = phase.repeat(num_slices, axis=0)
            data = numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase)
        else:
            data = existing_object_array

        return Object(
            array=data,
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
