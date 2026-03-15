import logging

import numpy.random

from ptychodus.api.probe_positions import ProbePositionSequence

from .builder import FromMemoryProbePositionsBuilder
from .builder_factory import ProbePositionsBuilderFactory
from .item import ProbePositionsRepositoryItem
from .settings import ProbePositionsSettings

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

    def create(
        self, position_seq: ProbePositionSequence | None = None
    ) -> ProbePositionsRepositoryItem:
        if position_seq is None:
            builder = self._builder_factory.create_default()
        else:
            builder = FromMemoryProbePositionsBuilder(self._rng, self._settings, position_seq)

        return ProbePositionsRepositoryItem(self._rng, self._settings, builder)

    def create_from_settings(self) -> ProbePositionsRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.exception(''.join(exc.args))
            builder = self._builder_factory.create_default()

        return ProbePositionsRepositoryItem(self._rng, self._settings, builder)
