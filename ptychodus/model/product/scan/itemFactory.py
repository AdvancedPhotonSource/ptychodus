import logging

import numpy

from ptychodus.api.scan import Scan

from .builder import FromMemoryScanBuilder
from .builderFactory import ScanBuilderFactory
from .item import ScanRepositoryItem
from .settings import ScanSettings
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 builderFactory: ScanBuilderFactory) -> None:
        self._rng = rng
        self._settings = settings
        self._builderFactory = builderFactory

    def create(self, scan: Scan | None = None) -> ScanRepositoryItem:
        builder = self._builderFactory.createDefault() if scan is None \
                else FromMemoryScanBuilder(scan)
        transform = ScanPointTransform(self._rng, self._settings)
        return ScanRepositoryItem(self._settings, builder, transform)

    def createFromSettings(self) -> ScanRepositoryItem:
        try:
            builder = self._builderFactory.createFromSettings()
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builderFactory.createDefault()

        transform = ScanPointTransform(self._rng, self._settings)
        return ScanRepositoryItem(self._settings, builder, transform)
