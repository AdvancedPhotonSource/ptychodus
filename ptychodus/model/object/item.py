from __future__ import annotations
import logging

from ...api.object import Object
from ...api.observer import Observable
from ...api.parametric import Parameter, ParameterRepository
from .builder import ObjectBuilder

logger = logging.getLogger(__name__)


class ObjectRepositoryItem(ParameterRepository):

    def __init__(self, name: Parameter[str], builder: ObjectBuilder) -> None:
        super().__init__('Object')
        self._name = name
        self._builder = builder
        self._object = Object()

        self._name.addObserver(self)
        self._addParameterRepository(builder, observe=True)

        self._rebuild()

    def getName(self) -> str:
        return self._name.getValue()

    def setName(self, name: str) -> None:
        self._name.setValue(name)

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
            object_ = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize object!')
            return

        self._object = object_
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._name:
            self.notifyObservers()
        elif observable is self._builder:
            self._rebuild()
        else:
            super().update(observable)
