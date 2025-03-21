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

        self.layer_distance_m = settings.object_layer_distance_m.copy()
        self._add_parameter('layer_distance_m', self.layer_distance_m)

        self._add_group('builder', builder, observe=True)
        self._rebuild()

    def assign_item(self, item: ObjectRepositoryItem) -> None:
        self.layer_distance_m.set_value(item.layer_distance_m.get_value(), notify=False)
        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, object_: Object) -> None:
        builder = FromMemoryObjectBuilder(self._settings, object_)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_num_layers(self) -> int:
        return len(self.layer_distance_m) + 1

    def set_num_layers(self, num_layers: int) -> None:
        num_spaces = max(0, num_layers - 1)
        distance_m = list(self.layer_distance_m.get_value())

        try:
            default_distance_m = distance_m[-1]
        except IndexError:
            default_distance_m = 0.0

        while len(distance_m) < num_spaces:
            distance_m.append(default_distance_m)

        if len(distance_m) > num_spaces:
            distance_m = distance_m[:num_spaces]

        self.layer_distance_m.set_value(distance_m)
        self._rebuild()

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
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            object_ = self._builder.build(
                self._geometry_provider, self.layer_distance_m.get_value()
            )
        except Exception as exc:
            logger.error(''.join(exc.args))
            return

        self._object = object_
        self.layer_distance_m.set_value(object_.layer_distance_m)
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super()._update(observable)
