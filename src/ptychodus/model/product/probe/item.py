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
        geometry_provider: ProbeGeometryProvider,
        settings: ProbeSettings,
        builder: ProbeBuilder,
        additional_modes_builder: MultimodalProbeBuilder | None,
    ) -> None:
        super().__init__()
        self._geometry_provider = geometry_provider
        self._settings = settings
        self._builder = builder
        self._additional_modes_builder = additional_modes_builder
        self._probe = Probe(array=None, pixel_geometry=None)

        self._add_group('builder', builder, observe=True)

        if additional_modes_builder is not None:
            self._add_group('additional_modes', additional_modes_builder, observe=True)

        self._rebuild()

    def assign_item(self, item: ProbeRepositoryItem) -> None:
        group = 'additional_modes'

        if self._additional_modes_builder is not None:
            self._remove_group(group)
            self._additional_modes_builder.remove_observer(self)
            self._additional_modes_builder = None

        additional_modes_builder = item.get_additional_modes_builder()

        if additional_modes_builder is not None:
            self._additional_modes_builder = additional_modes_builder.copy()
            self._additional_modes_builder.add_observer(self)
            self._add_group(group, self._additional_modes_builder, observe=True)

        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, probe: Probe) -> None:
        builder = FromMemoryProbeBuilder(self._settings, probe)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

        if self._additional_modes_builder is not None:
            self._additional_modes_builder.sync_to_settings()

    def get_probe(self) -> Probe:
        return self._probe

    def get_builder(self) -> ProbeBuilder:
        return self._builder

    def set_builder(self, builder: ProbeBuilder) -> None:
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
        except Exception as exc:
            logger.exception('Failed to rebuild probe!')
            return

        self._probe = (
            probe
            if self._additional_modes_builder is None
            else self._additional_modes_builder.build(probe, self._geometry_provider)
        )
        self.notify_observers()

    def get_additional_modes_builder(self) -> MultimodalProbeBuilder | None:
        return self._additional_modes_builder

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._additional_modes_builder:
            self._rebuild()
        else:
            super()._update(observable)
