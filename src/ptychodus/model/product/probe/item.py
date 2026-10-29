from __future__ import annotations
import logging

import numpy

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.probe import (
    ProbeEntropyMetrics,
    ProbeGeometryProvider,
    ProbeEnsemble,
    ProbeSizeMetrics,
    estimate_probe_entropy,
    estimate_probe_size,
)

from ...observer import Observable
from ...parameters import ParameterGroup
from .builder import FromMemoryProbeBuilder, ProbeEnsembleBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItem(ParameterGroup):
    def __init__(
        self,
        rng: numpy.random.Generator,
        geometry_provider: ProbeGeometryProvider,
        settings: ProbeSettings,
        builder: ProbeEnsembleBuilder,
    ) -> None:
        super().__init__()
        self._rng = rng
        self._geometry_provider = geometry_provider
        self._settings = settings
        self._builder = builder
        self._probe_seq = ProbeEnsemble(array=None, opr_weights=None, pixel_geometry=None)

        self._add_group('builder', builder, observe=True)
        if isinstance(geometry_provider, Observable):
            geometry_provider.add_observer(self)
        self._rebuild()

    def assign_item(self, item: ProbeRepositoryItem) -> None:
        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, probe: ProbeEnsemble) -> None:
        builder = FromMemoryProbeBuilder(self._rng, self._settings, probe)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_probes(self) -> ProbeEnsemble:
        return self._probe_seq

    def get_num_scan_points(self) -> int:
        """Number of probe positions this probe is being built against.

        Named after the geometry-provider property it forwards.
        """
        return self._geometry_provider.num_scan_points

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

    def can_estimate_size_from_diffraction(self) -> bool:
        """Whether the builder has a size to estimate and the product has data to estimate it from."""
        return self._builder.can_estimate_size_from_diffraction(self._geometry_provider)

    def estimate_size_from_diffraction(self) -> bool:
        """Set the builder's size parameters from the measured patterns; see the builder."""
        return self._builder.estimate_size_from_diffraction(self._geometry_provider)

    def handle_diffraction_data_changed(self) -> None:
        """Rebuild if the builder reads the diffraction data, which has just changed."""
        if self._builder.reads_diffraction_data:
            self._rebuild()

    def get_builder(self) -> ProbeEnsembleBuilder:
        return self._builder

    def can_make_editable(self) -> bool:
        """Whether the probe is a result whose mode structure cannot yet be extended."""
        return isinstance(self._builder, FromMemoryProbeBuilder) and not self._builder.is_editable

    def make_editable(self) -> None:
        """Let a result take on more modes; see :meth:`FromMemoryProbeBuilder.make_editable`."""
        if isinstance(self._builder, FromMemoryProbeBuilder):
            self._builder.make_editable()

    def set_builder(self, builder: ProbeEnsembleBuilder) -> None:
        group = 'builder'
        self._remove_group(group)
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group(group, self._builder, observe=True)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            probe_seq = self._builder.build(self._geometry_provider)
        except GeometryNotDefinedError:
            # Not yet bound; the observer wired in __init__ re-runs this once the
            # geometry is determined. Must precede the catch-all below, or a routine
            # startup state would be logged as a failure.
            return
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
