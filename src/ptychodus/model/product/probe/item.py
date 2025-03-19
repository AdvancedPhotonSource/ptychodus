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
        self._probe = Probe(array=None, pixel_geometry=None)

        self._add_group('builder', builder, observe=True)
        self._add_group('additional_modes', additionalModesBuilder, observe=True)

        self._rebuild()

    def assign_item(self, item: ProbeRepositoryItem) -> None:
        self._remove_group('additional_modes')
        self._additionalModesBuilder.remove_observer(self)
        self._additionalModesBuilder = item.getAdditionalModesBuilder().copy()
        self._additionalModesBuilder.add_observer(self)
        self._add_group('additional_modes', self._additionalModesBuilder, observe=True)

        self.setBuilder(item.getBuilder().copy())
        self._rebuild()

    def assign(self, probe: Probe, *, mutable: bool = True) -> None:
        builder = FromMemoryProbeBuilder(self._settings, probe)
        self.setBuilder(builder, mutable=mutable)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.syncToSettings()
        self._additionalModesBuilder.syncToSettings()

    def get_probe(self) -> Probe:
        return self._probe

    def getBuilder(self) -> ProbeBuilder:
        return self._builder

    def setBuilder(self, builder: ProbeBuilder, *, mutable: bool = True) -> None:
        self._remove_group('builder')
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group('builder', self._builder, observe=True)
        self._rebuild(mutable=mutable)

    def _rebuild(self, *, mutable: bool = True) -> None:
        try:
            probe = self._builder.build(self._geometryProvider)
        except Exception as exc:
            logger.error(''.join(exc.args))
            return

        self._probe = (
            self._additionalModesBuilder.build(probe, self._geometryProvider) if mutable else probe
        )
        self.notify_observers()

    def getAdditionalModesBuilder(self) -> MultimodalProbeBuilder:
        return self._additionalModesBuilder

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._additionalModesBuilder:
            self._rebuild()
        else:
            super()._update(observable)
