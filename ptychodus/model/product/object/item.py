from __future__ import annotations
import logging

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterRepository

from .builder import ObjectBuilder

logger = logging.getLogger(__name__)


class ObjectRepositoryItem(ParameterRepository):

    def __init__(self, geometryProvider: ObjectGeometryProvider, builder: ObjectBuilder) -> None:
        super().__init__('Object')
        self._geometryProvider = geometryProvider
        self._builder = builder
        self._object = Object()

        self._addParameterRepository(builder, observe=True)
        self.layerDistanceInMeters = self._registerRealArrayParameter(
            'layer_distance_m', [numpy.inf])

        self._rebuild()

    def assign(self, item: ObjectRepositoryItem) -> None:
        self.layerDistanceInMeters.setValue(item.layerDistanceInMeters.getValue(), notify=False)
        self.setBuilder(item.getBuilder().copy())

    def getObject(self) -> Object:
        return self._object

    def getBuilder(self) -> ObjectBuilder:
        return self._builder

    def setBuilder(self, builder: ObjectBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder, observe=True)
        self._rebuild()

    def _rebuild(self) -> None:
        layerDistanceInMeters = list(self.layerDistanceInMeters.getValue())

        if len(layerDistanceInMeters) < 1:
            layerDistanceInMeters.append(numpy.inf)

        try:
            object_ = self._builder.build(self._geometryProvider, layerDistanceInMeters)
        except Exception:
            logger.exception('Failed to reinitialize object!')
            return

        self._object = object_
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super().update(observable)
