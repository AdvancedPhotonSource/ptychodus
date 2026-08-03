from __future__ import annotations
import logging

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import (
    ProbeEntropyMetrics,
    ProbeGeometryProvider,
    ProbeSequence,
    ProbeSizeMetrics,
    estimate_probe_entropy,
    estimate_probe_size,
)

from .builder import FromMemoryProbeBuilder, ProbeSequenceBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterGroup):
    def __init__(
        self,
        geometry_provider: ProbeGeometryProvider,
        settings: ProbeSettings,
        builder: ProbeSequenceBuilder,
    ) -> None:
        super().__init__()
        self._geometry_provider = geometry_provider
        self._settings = settings
        self._builder = builder
        self._probe_seq = ProbeSequence(array=None, opr_weights=None, pixel_geometry=None)

        self._add_group('builder', builder, observe=True)
        if isinstance(geometry_provider, Observable):
            geometry_provider.add_observer(self)
        self._rebuild()

    def assign_item(self, item: ProbeRepositoryItem) -> None:
        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, probe: ProbeSequence) -> None:
        builder = FromMemoryProbeBuilder(self._settings, probe)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_probes(self) -> ProbeSequence:
        return self._probe_seq

    def get_size_metrics(self) -> ProbeSizeMetrics | None:
        probe_seq = self._probe_seq

        if probe_seq.width_px == 0 or probe_seq.height_px == 0:
            return None

        try:
            pixel_geometry = probe_seq.get_pixel_geometry()
            probe = probe_seq.get_probe_no_opr()
        except ValueError:
            return None

        try:
            return estimate_probe_size(probe.get_intensity(), pixel_geometry)
        except Exception:
            logger.exception('Failed to estimate probe size!')
            return None

    def get_entropy_metrics(self) -> ProbeEntropyMetrics | None:
        probe_seq = self._probe_seq

        if probe_seq.width_px == 0 or probe_seq.height_px == 0:
            return None

        try:
            probe = probe_seq.get_probe_no_opr()
        except ValueError:
            return None

        try:
            return estimate_probe_entropy(probe)
        except Exception:
            logger.exception('Failed to estimate probe entropy!')
            return None

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
        if not self._geometry_provider.get_probe_geometry().get_pixel_geometry().is_valid:
            # Geometry not yet bound; the observer wired in __init__ will re-run
            # _rebuild when the geometry becomes valid.
            return
        try:
            probe_seq = self._builder.build(self._geometry_provider)
        except Exception:
            logger.exception('Failed to rebuild probe!')
        else:
            self._probe_seq = probe_seq
            self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._geometry_provider:
            self._rebuild()
        else:
            super()._update(observable)
