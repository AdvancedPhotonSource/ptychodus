import logging

import numpy.random

from ptychodus.api.scan import PositionSequence

from .builder import FromMemoryScanBuilder
from .builder_factory import ScanBuilderFactory
from .item import ScanRepositoryItem
from .settings import ScanSettings
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ScanSettings,
        builder_factory: ScanBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builder_factory = builder_factory

    def create(self, scan: PositionSequence | None = None) -> ScanRepositoryItem:
        transform = ScanPointTransform(self._rng, self._settings)

        if scan is None:
            builder = self._builder_factory.create_default()
        else:
            builder = FromMemoryScanBuilder(self._settings, scan)
            transform.set_identity()

        return ScanRepositoryItem(self._settings, builder, transform)

    def create_from_settings(self) -> ScanRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.exception(''.join(exc.args))
            builder = self._builder_factory.create_default()

        transform = ScanPointTransform(self._rng, self._settings)
        return ScanRepositoryItem(self._settings, builder, transform)
