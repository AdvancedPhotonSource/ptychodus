from __future__ import annotations
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder, ProbeSequenceBuilder
from .multimodal import MultimodalProbeBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterGroup):
    def __init__(
        self,
        geometry_provider: ProbeGeometryProvider,
        settings: ProbeSettings,
        builder: ProbeSequenceBuilder,
        additional_modes_builder: MultimodalProbeBuilder,
    ) -> None:
        super().__init__()
        self._geometry_provider = geometry_provider
        self._settings = settings
        self._builder = builder
        self._additional_modes_builder = additional_modes_builder
        self._probe_seq = ProbeSequence(array=None, opr_weights=None, pixel_geometry=None)

        self._add_group('builder', builder, observe=True)
        self._add_group('additional_modes', additional_modes_builder, observe=True)

        self._rebuild()

    def assign_item(self, item: ProbeRepositoryItem) -> None:
        group = 'additional_modes'

        self._remove_group(group)
        self._additional_modes_builder.remove_observer(self)

        additional_modes_builder = item.get_additional_modes_builder()

        self._additional_modes_builder = additional_modes_builder.copy()
        self._additional_modes_builder.add_observer(self)
        self._add_group(group, self._additional_modes_builder, observe=True)

        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, probe: ProbeSequence) -> None:
        builder = FromMemoryProbeBuilder(self._settings, probe)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()
        self._additional_modes_builder.sync_to_settings()

    def get_probes(self) -> ProbeSequence:
        return self._probe_seq

    def get_builder(self) -> ProbeSequenceBuilder:
        return self._builder

    def set_builder(self, builder: ProbeSequenceBuilder) -> None:
        group = 'builder'
        self._remove_group(group)
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group(group, self._builder, observe=True)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            probe = self._builder.build(self._geometry_provider)
        except Exception:
            logger.exception('Failed to rebuild probe!')
            return

        self._probe_seq = self._additional_modes_builder.build(probe, self._geometry_provider)
        self.notify_observers()

    def get_additional_modes_builder(self) -> MultimodalProbeBuilder:
        return self._additional_modes_builder

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._additional_modes_builder:
            self._rebuild()
        else:
            super()._update(observable)
