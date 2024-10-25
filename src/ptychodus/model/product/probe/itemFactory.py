import logging

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import FromMemoryProbeBuilder
from .builderFactory import ProbeBuilderFactory
from .item import ProbeRepositoryItem
from .multimodal import MultimodalProbeBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        builderFactory: ProbeBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builderFactory = builderFactory

    def create(
        self, geometryProvider: ProbeGeometryProvider, probe: Probe | None = None
    ) -> ProbeRepositoryItem:
        builder = (
            self._builderFactory.createDefault()
            if probe is None
            else FromMemoryProbeBuilder(self._settings, probe)
        )
        multimodalBuilder = MultimodalProbeBuilder(self._rng, self._settings)
        return ProbeRepositoryItem(geometryProvider, self._settings, builder, multimodalBuilder)

    def createFromSettings(self, geometryProvider: ProbeGeometryProvider) -> ProbeRepositoryItem:
        try:
            builder = self._builderFactory.createFromSettings()
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builderFactory.createDefault()

        multimodalBuilder = MultimodalProbeBuilder(self._rng, self._settings)
        return ProbeRepositoryItem(geometryProvider, self._settings, builder, multimodalBuilder)
