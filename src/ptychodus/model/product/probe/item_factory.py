import logging

import numpy.random

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder
from .builder_factory import ProbeBuilderFactory
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        builder_factory: ProbeBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builder_factory = builder_factory

    def create(
        self, geometry_provider: ProbeGeometryProvider, probe: ProbeSequence | None = None
    ) -> ProbeRepositoryItem:
        multimodal_builder = MultimodalProbeBuilder(self._rng, self._settings)

        if probe is None:
            builder = self._builder_factory.create_default()
        else:
            builder = FromMemoryProbeBuilder(self._settings, probe)
            multimodal_builder.set_identity()

        return ProbeRepositoryItem(geometry_provider, self._settings, builder, multimodal_builder)

    def create_from_settings(self, geometry_provider: ProbeGeometryProvider) -> ProbeRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builder_factory.create_default()

        multimodal_builder = MultimodalProbeBuilder(self._rng, self._settings)
        return ProbeRepositoryItem(geometry_provider, self._settings, builder, multimodal_builder)
