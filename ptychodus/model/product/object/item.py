from __future__ import annotations
import logging

import numpy

from ptychodus.api.object import Object
from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterRepository

from .builder import ObjectBuilder

logger = logging.getLogger(__name__)


class ObjectRepositoryItem(ParameterRepository):

    def __init__(self, builder: ObjectBuilder) -> None:
        super().__init__('Object')
        self._builder = builder
        self._object = Object()
        self._layerDistanceInMeters: list[float] = list()

        self._addParameterRepository(builder, observe=True)

        self._rebuild()

    def getObject(self) -> Object:
        return self._object

    def getBuilder(self) -> ObjectBuilder:
        return self._builder

    def setBuilder(self, builder: ObjectBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            object_ = self._builder.build()  # FIXME (self._layerDistanceInMeters)
        except Exception:
            logger.exception('Failed to reinitialize object!')
            return

        self._object = object_
        self._layerDistanceInMeters = list(object_.layerDistanceInMeters)
        self.notifyObservers()

    def getLayerDistanceInMeters(self, number: int) -> float:
        return self._layerDistanceInMeters[number]

    def setLayerDistanceInMeters(self, number: int, distanceInMeters: float) -> None:
        if numpy.isfinite(distanceInMeters) and distanceInMeters > 0.:
            if self._layerDistanceInMeters[number] != distanceInMeters:
                self._layerDistanceInMeters[number] = distanceInMeters
                self._rebuild()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super().update(observable)
