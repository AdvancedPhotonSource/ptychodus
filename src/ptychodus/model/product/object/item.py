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
        geometryProvider: ObjectGeometryProvider,
        settings: ObjectSettings,
        builder: ObjectBuilder,
    ) -> None:
        super().__init__()
        self._geometryProvider = geometryProvider
        self._settings = settings
        self._builder = builder
        self._object = Object(array=None, pixel_geometry=None, center=None)

        self.layerDistanceInMeters = settings.objectLayerDistanceInMeters.copy()
        self._add_parameter('layer_distance_m', self.layerDistanceInMeters)

        self._add_group('builder', builder, observe=True)
        self._rebuild()

    def assignItem(self, item: ObjectRepositoryItem) -> None:
        self.layerDistanceInMeters.set_value(item.layerDistanceInMeters.get_value(), notify=False)
        self.setBuilder(item.getBuilder().copy())
        self._rebuild()

    def assign(self, object_: Object, *, mutable: bool = True) -> None:
        builder = FromMemoryObjectBuilder(self._settings, object_)
        self.setBuilder(builder, mutable=mutable)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.syncToSettings()

    def getNumberOfLayers(self) -> int:
        return len(self.layerDistanceInMeters) + 1

    def setNumberOfLayers(self, numberOfLayers: int) -> None:
        numberOfSpaces = max(0, numberOfLayers - 1)
        distanceInMeters = list(self.layerDistanceInMeters.get_value())

        try:
            defaultDistanceInMeters = distanceInMeters[-1]
        except IndexError:
            defaultDistanceInMeters = 0.0

        while len(distanceInMeters) < numberOfSpaces:
            distanceInMeters.append(defaultDistanceInMeters)

        if len(distanceInMeters) > numberOfSpaces:
            distanceInMeters = distanceInMeters[:numberOfSpaces]

        self.layerDistanceInMeters.set_value(distanceInMeters)
        self._rebuild()

    def get_object(self) -> Object:
        return self._object

    def getBuilder(self) -> ObjectBuilder:
        return self._builder

    def setBuilder(self, builder: ObjectBuilder, *, mutable: bool = True) -> None:
        self._remove_group('builder')
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group('builder', self._builder, observe=True)
        self._rebuild(mutable=mutable)

    def _rebuild(self, *, mutable: bool = True) -> None:
        try:
            object_ = self._builder.build(
                self._geometryProvider, self.layerDistanceInMeters.get_value()
            )
        except Exception as exc:
            logger.error(''.join(exc.args))
            return

        # TODO mutable is unused
        self._object = object_
        self.layerDistanceInMeters.set_value(object_.layer_distance_m)
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super()._update(observable)
