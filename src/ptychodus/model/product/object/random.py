from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.model.phaseUnwrapper import PhaseUnwrapper

from .builder import ObjectBuilder
from .settings import ObjectSettings


class RandomObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'random')
        self._rng = rng
        self._settings = settings

        self.extraPaddingX = settings.extraPaddingX.copy()
        self._add_parameter('extra_padding_x', self.extraPaddingX)
        self.extraPaddingY = settings.extraPaddingY.copy()
        self._add_parameter('extra_padding_y', self.extraPaddingY)

        self.amplitudeMean = settings.amplitudeMean.copy()
        self._add_parameter('amplitude_mean', self.amplitudeMean)
        self.amplitudeDeviation = settings.amplitudeDeviation.copy()
        self._add_parameter('amplitude_deviation', self.amplitudeDeviation)

        self.phaseDeviation = settings.phaseDeviation.copy()
        self._add_parameter('phase_deviation', self.phaseDeviation)

    def copy(self) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        geometry = geometryProvider.get_object_geometry()
        heightInPixels = geometry.height_px + 2 * self.extraPaddingY.get_value()
        widthInPixels = geometry.width_px + 2 * self.extraPaddingX.get_value()
        objectShape = (1 + len(layerDistanceInMeters), heightInPixels, widthInPixels)

        amplitude = self._rng.normal(
            self.amplitudeMean.get_value(),
            self.amplitudeDeviation.get_value(),
            objectShape,
        )
        phase = self._rng.normal(
            0.0,
            self.phaseDeviation.get_value(),
            objectShape,
        )

        return Object(
            array=numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase),
            layer_distance_m=layerDistanceInMeters,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )


class UserObjectBuilder(ObjectBuilder):  # FIXME use
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
        self._existingObject = object_
        self._settings = settings

    def copy(self) -> UserObjectBuilder:
        builder = UserObjectBuilder(self._existingObject, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometryProvider: ObjectGeometryProvider,
        layerDistanceInMeters: Sequence[float],
    ) -> Object:
        geometry = self._existingObject.get_geometry()
        exitingObjectArr = self._existingObject.get_array()
        nSlices = len(layerDistanceInMeters) + 1

        if nSlices > 1 and nSlices != exitingObjectArr.shape[0]:
            amplitude = numpy.abs(exitingObjectArr[0:1]) ** (1.0 / nSlices)
            amplitude = amplitude.repeat(nSlices, axis=0)
            phase = PhaseUnwrapper().unwrap(exitingObjectArr[0])[None, ...] / nSlices
            phase = phase.repeat(nSlices, axis=0)
            data = numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase)
        else:
            data = exitingObjectArr

        return Object(
            array=data,
            layer_distance_m=layerDistanceInMeters,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
