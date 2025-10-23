import logging

import numpy.random

from ptychodus.api.probe_positions import ProbePositionSequence

from .builder import FromMemoryProbePositionsBuilder
from .builder_factory import ProbePositionsBuilderFactory
from .item import ProbePositionsRepositoryItem
from .settings import ProbePositionsSettings
from .transform import ProbePositionTransform

logger = logging.getLogger(__name__)


class ProbePositionsRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        builder_factory: ProbePositionsBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builder_factory = builder_factory

    def create(self, scan: ProbePositionSequence | None = None) -> ProbePositionsRepositoryItem:
        transform = ProbePositionTransform(self._rng, self._settings)

        if scan is None:
            builder = self._builder_factory.create_default()
        else:
            builder = FromMemoryProbePositionsBuilder(self._settings, scan)
            transform.set_identity()

        return ProbePositionsRepositoryItem(self._settings, builder, transform)

    def create_from_settings(self) -> ProbePositionsRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.exception(''.join(exc.args))
            builder = self._builder_factory.create_default()

        transform = ProbePositionTransform(self._rng, self._settings)
        return ProbePositionsRepositoryItem(self._settings, builder, transform)
