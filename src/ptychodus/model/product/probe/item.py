from __future__ import annotations
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder, ProbeBuilder
from .multimodal import MultimodalProbeBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterGroup):
    def __init__(
        self,
        geometryProvider: ProbeGeometryProvider,
        settings: ProbeSettings,
        builder: ProbeBuilder,
        additionalModesBuilder: MultimodalProbeBuilder,
    ) -> None:
        super().__init__()
        self._geometryProvider = geometryProvider
        self._settings = settings
        self._builder = builder
        self._additionalModesBuilder = additionalModesBuilder
        self._probe = Probe()

        self._addGroup('builder', builder, observe=True)
        self._addGroup('additional_modes', additionalModesBuilder, observe=True)

        self._rebuild()

    def assignItem(self, item: ProbeRepositoryItem) -> None:
        self._removeGroup('additional_modes')
        self._additionalModesBuilder.removeObserver(self)
        self._additionalModesBuilder = item.getAdditionalModesBuilder().copy()
        self._additionalModesBuilder.addObserver(self)
        self._addGroup('additional_modes', self._additionalModesBuilder, observe=True)

        self.setBuilder(item.getBuilder().copy())
        self._rebuild()

    def assign(self, probe: Probe) -> None:
        builder = FromMemoryProbeBuilder(self._settings, probe)
        self.setBuilder(builder)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

        self._builder.syncToSettings()
        self._additionalModesBuilder.syncToSettings()

    def getProbe(self) -> Probe:
        return self._probe

    def getBuilder(self) -> ProbeBuilder:
        return self._builder

    def setBuilder(self, builder: ProbeBuilder) -> None:
        self._removeGroup('builder')
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addGroup('builder', self._builder, observe=True)

    def _rebuild(self) -> None:
        try:
            probe = self._builder.build(self._geometryProvider)
        except Exception as exc:
            logger.error(''.join(exc.args))
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
