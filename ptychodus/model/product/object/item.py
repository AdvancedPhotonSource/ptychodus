from __future__ import annotations
import logging

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup, RealArrayParameter

from .builder import ObjectBuilder
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
        self._object = Object()

        self._addGroup('builder', builder, observe=True)
        self.layerDistanceInMeters = RealArrayParameter(self, 'layer_distance_m', [numpy.inf])

        self._rebuild()

    def assign(self, item: ObjectRepositoryItem) -> None:
        self.layerDistanceInMeters.setValue(item.layerDistanceInMeters.getValue(), notify=False)
        self.setBuilder(item.getBuilder().copy())

    def getNumberOfLayers(self) -> int:
        return len(self.layerDistanceInMeters)

    def setNumberOfLayers(self, number: int) -> None:
        numRequested = max(1, number)
        distanceInMeters = list(self.layerDistanceInMeters.getValue())
        numExisting = len(distanceInMeters)
        defaultDistanceInMeters = float(self._settings.objectLayerDistanceInMeters.getValue())

        if numExisting < 2:
            distanceInMeters = [defaultDistanceInMeters] * numRequested
        elif numExisting < numRequested:
            distanceInMeters[-1] = distanceInMeters[-2]  # overwrite inf
            distanceInMeters.extend(distanceInMeters[-1:] * (numRequested - numExisting))
        elif numExisting > numRequested:
            distanceInMeters = distanceInMeters[:numRequested]

        distanceInMeters[-1] = numpy.inf
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
        layerDistanceInMeters = list(self.layerDistanceInMeters.getValue())

        if len(layerDistanceInMeters) < 1:
            layerDistanceInMeters.append(numpy.inf)

        try:
            object_ = self._builder.build(self._geometryProvider, layerDistanceInMeters)
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
