from __future__ import annotations
import logging

from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.probe import Probe
from .builder import ProbeBuilder
from .multimodal import MultimodalProbeBuilder

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterRepository):

    def __init__(self, builder: ProbeBuilder, modesBuilder: MultimodalProbeBuilder) -> None:
        super().__init__('Probe')
        self._builder = builder
        self._modesBuilder = modesBuilder
        self._probe = Probe()

        self._addParameterRepository(builder, observe=True)
        self._addParameterRepository(modesBuilder, observe=True)

        self._rebuild()

    def getProbe(self) -> Probe:
        return self._probe

    def getBuilder(self) -> ProbeBuilder:
        return self._builder

    def setBuilder(self, builder: ProbeBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            probe = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize probe!')
            return

        self._probe = self._modesBuilder.build(probe)

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._modesBuilder:
            self._rebuild()
        else:
            super().update(observable)
