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
        self._object = Object(array=None, pixelGeometry=None, center=None)

        self.layerDistanceInMeters = settings.objectLayerDistanceInMeters.copy()
        self._addParameter('layer_distance_m', self.layerDistanceInMeters)

        self._addGroup('builder', builder, observe=True)
        self._rebuild()

    def assignItem(self, item: ObjectRepositoryItem) -> None:
        self.layerDistanceInMeters.setValue(item.layerDistanceInMeters.getValue(), notify=False)
        self.setBuilder(item.getBuilder().copy())
        self._rebuild()

    def assign(self, object_: Object) -> None:
        builder = FromMemoryObjectBuilder(self._settings, object_)
        self.setBuilder(builder)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

        self._builder.syncToSettings()

    def getNumberOfLayers(self) -> int:
        return len(self.layerDistanceInMeters) + 1

    def setNumberOfLayers(self, numberOfLayers: int) -> None:
        numberOfSpaces = max(0, numberOfLayers - 1)
        distanceInMeters = list(self.layerDistanceInMeters.getValue())

        try:
            defaultDistanceInMeters = distanceInMeters[-1]
        except IndexError:
            defaultDistanceInMeters = 0.0

        while len(distanceInMeters) < numberOfSpaces:
            distanceInMeters.append(defaultDistanceInMeters)

        if len(distanceInMeters) > numberOfSpaces:
            distanceInMeters = distanceInMeters[:numberOfSpaces]

        self.layerDistanceInMeters.setValue(distanceInMeters)
        self._rebuild()

    def getObject(self) -> Object:
        return self._object

    def getBuilder(self) -> ObjectBuilder:
        return self._builder

    def setBuilder(self, builder: ObjectBuilder) -> None:
        self._removeGroup('builder')
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addGroup('builder', self._builder, observe=True)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            object_ = self._builder.build(
                self._geometryProvider, self.layerDistanceInMeters.getValue()
            )
        except Exception as exc:
            logger.error(''.join(exc.args))
            return

        self._object = object_
        self.layerDistanceInMeters.setValue(object_.layerDistanceInMeters)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super().update(observable)
