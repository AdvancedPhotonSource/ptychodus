from __future__ import annotations
import logging
import math


from ptychodus.api.constants import format_length
from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.parameters import ParameterGroup

from .builder import FromMemoryObjectBuilder, ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectRepositoryItem(ParameterGroup):
    def __init__(
        self,
        geometry_provider: ObjectGeometryProvider,
        settings: ObjectSettings,
        builder: ObjectBuilder,
    ) -> None:
        super().__init__()
        self._geometry_provider = geometry_provider
        self._settings = settings
        self._builder = builder
        self._object = Object(array=None, pixel_geometry=None, center=None)

        self.layer_spacing_m = settings.object_layer_spacing_m.copy()
        self._add_parameter('layer_spacing_m', self.layer_spacing_m)

        self._add_group('builder', builder, observe=True)
        if isinstance(geometry_provider, Observable):
            geometry_provider.add_observer(self)
        self.rebuild()

    def assign_item(self, item: ObjectRepositoryItem) -> None:
        self.layer_spacing_m.set_value(item.layer_spacing_m.get_value(), notify=False)
        self.set_builder(item.get_builder().copy())
        self.rebuild()

    def assign(self, object_: Object) -> None:
        builder = FromMemoryObjectBuilder(self._settings, object_)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_num_layers(self) -> int:
        return len(self.layer_spacing_m) + 1

    def set_num_layers(self, num_layers: int) -> None:
        num_spaces = max(0, num_layers - 1)
        distance_m = list(self.layer_spacing_m.get_value())

        try:
            default_distance_m = distance_m[-1]
        except IndexError:
            default_distance_m = 0.0

        while len(distance_m) < num_spaces:
            distance_m.append(default_distance_m)

        if len(distance_m) > num_spaces:
            distance_m = distance_m[:num_spaces]

        self.layer_spacing_m.set_value(distance_m)
        self.rebuild()

    def get_object(self) -> Object:
        return self._object

    def get_builder(self) -> ObjectBuilder:
        return self._builder

    def set_builder(self, builder: ObjectBuilder) -> None:
        self._remove_group('builder')
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group('builder', self._builder, observe=True)
        self.rebuild()

    def rebuild(self, *, recenter: bool = False) -> None:
        try:
            object_ = self._builder.build(self._geometry_provider, self.layer_spacing_m.get_value())
        except GeometryNotDefinedError:
            # Not yet bound; the observer wired in __init__ re-runs this once the
            # geometry is determined. Must precede the catch-all below, or a routine
            # startup state would be logged as a failure.
            return
        except Exception:
            logger.exception('Failed to rebuild object!')
            return

        _warn_if_pixel_size_disagrees(object_, self._geometry_provider, self._builder.get_name())

        if recenter:
            object_geometry = self._geometry_provider.get_object_geometry()
            self._object = Object(
                array=object_.get_array(),
                layer_spacing_m=object_.layer_spacing_m,
                pixel_geometry=object_.get_pixel_geometry(),
                center=object_geometry.get_center(),
            )
        else:
            self._object = object_

        self.layer_spacing_m.set_value(object_.layer_spacing_m)
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self.rebuild()
        elif observable is self._geometry_provider:
            self.rebuild()
        else:
            super()._update(observable)


def _warn_if_pixel_size_disagrees(
    object_: Object,
    geometry_provider: ObjectGeometryProvider,
    builder_name: str,
    *,
    rel_tol: float = 1.0e-9,
) -> None:
    """Warn when a built object is sampled differently from the geometry it will run against.

    A from-file object is reconciled by its builder, but an object that arrives already
    conditioned keeps whatever sampling it was saved with: rebinding a product to another
    dataset, or editing the photon energy or detector distance, moves the run's pixel size
    out from under it. Resampling here is the wrong remedy -- the same path carries
    reconstruction output, reassigned on every iteration, and a product read back from
    file is a finished record rather than an initial guess -- so the disagreement is
    reported and the array left alone. Output products agree within ``rel_tol``, so this
    stays silent for them.
    """
    try:
        pixel_geometry = object_.get_pixel_geometry()
        # GeometryNotDefinedError is a ValueError, so an unbound provider lands here
        # alongside an object that never recorded its own pixel size.
        expected = geometry_provider.get_object_geometry().get_pixel_geometry()
    except ValueError:
        return

    if math.isclose(pixel_geometry.width_m, expected.width_m, rel_tol=rel_tol) and math.isclose(
        pixel_geometry.height_m, expected.height_m, rel_tol=rel_tol
    ):
        return

    logger.warning(
        'Object from builder "%s" is sampled at %s x %s but this product samples at '
        '%s x %s; it will reconstruct at the wrong scale.',
        builder_name,
        format_length(pixel_geometry.width_m),
        format_length(pixel_geometry.height_m),
        format_length(expected.width_m),
        format_length(expected.height_m),
    )
