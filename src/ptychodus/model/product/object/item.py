from __future__ import annotations
import logging


from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup

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
        except Exception:
            logger.exception('Failed to rebuild object!')
            return

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
        else:
            super()._update(observable)
