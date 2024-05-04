from __future__ import annotations
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder
from .multimodal import MultimodalProbeBuilder

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterRepository):

    def __init__(self, geometryProvider: ProbeGeometryProvider, builder: ProbeBuilder,
                 additionalModesBuilder: MultimodalProbeBuilder) -> None:
        super().__init__('Probe')
        self._geometryProvider = geometryProvider
        self._builder = builder
        self._additionalModesBuilder = additionalModesBuilder
        self._probe = Probe()

        self._addParameterRepository(builder, observe=True)
        self._addParameterRepository(additionalModesBuilder, observe=True)

        self._rebuild()

    def assign(self, item: ProbeRepositoryItem) -> None:
        self._removeParameterRepository(self._additionalModesBuilder)
        self._additionalModesBuilder.removeObserver(self)
        self._additionalModesBuilder = item.getAdditionalModesBuilder().copy()
        self._additionalModesBuilder.addObserver(self)
        self._addParameterRepository(self._additionalModesBuilder, observe=True)

        self.setBuilder(item.getBuilder().copy())

    def getProbe(self) -> Probe:
        return self._probe

    def getBuilder(self) -> ProbeBuilder:
        return self._builder

    def setBuilder(self, builder: ProbeBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder, observe=True)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            probe = self._builder.build(self._geometryProvider)
        except Exception:
            logger.exception('Failed to reinitialize probe!')
            return

        self._probe = self._additionalModesBuilder.build(probe)
        self.notifyObservers()

    def getAdditionalModesBuilder(self) -> MultimodalProbeBuilder:
        return self._additionalModesBuilder

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._additionalModesBuilder:
            self._rebuild()
        else:
            super().update(observable)
