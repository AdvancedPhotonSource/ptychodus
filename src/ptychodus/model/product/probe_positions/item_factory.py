import logging

import numpy.random

from ptychodus.api.probe_positions import ProbePositionSequence

from .builder import FromMemoryProbePositionsBuilder
from .builder_factory import ProbePositionsBuilderFactory
from .item import ProbePositionsRepositoryItem
from .settings import ProbePositionsSettings
from ..metadata import MetadataRepositoryItem

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

    def _warn_if_conditioning_ignored(self) -> None:
        """Note that conditioning settings do not apply to in-memory positions.

        Positions supplied in memory come from reconstruction output or a product
        loaded from file, so they are already conditioned. Batch mode reads
        product.h5 through this path, where a user who sets a trim in
        settings.ini would otherwise see it silently do nothing. Trim at ingest
        instead -- ptychodus-bdp reads raw probe positions through the from-file
        builder, which does condition them.
        """
        settings = self._settings
        is_conditioning_requested = (
            settings.num_discard_at_start.get_value() != 0
            or settings.num_discard_at_end.get_value() != 0
            or settings.jitter_radius_m.get_value() != 0.0
            or settings.affine00.get_value() != 1.0
            or settings.affine01.get_value() != 0.0
            or settings.affine02.get_value() != 0.0
            or settings.affine10.get_value() != 0.0
            or settings.affine11.get_value() != 1.0
            or settings.affine12.get_value() != 0.0
        )

        if is_conditioning_requested:
            logger.info(
                'Probe positions supplied in memory are already conditioned;'
                ' ignoring the trim, affine transform, and jitter settings.'
            )

    def create(
        self,
        position_seq: ProbePositionSequence | None = None,
        *,
        metadata_item: MetadataRepositoryItem | None = None,
    ) -> ProbePositionsRepositoryItem:
        """An item over the default builder, or over *position_seq* in the exit frame.

        *position_seq* is in the exit frame of *metadata_item*'s geometry, as a product's
        positions are, and is taken back to the lab frame once.
        """
        if position_seq is None:
            return ProbePositionsRepositoryItem(
                self._rng,
                self._settings,
                self._builder_factory.create_default(),
                metadata_item=metadata_item,
            )

        self._warn_if_conditioning_ignored()
        item = ProbePositionsRepositoryItem(
            self._rng,
            self._settings,
            FromMemoryProbePositionsBuilder(self._rng, self._settings, ProbePositionSequence()),
            metadata_item=metadata_item,
        )
        item.assign(position_seq)
        return item

    def create_from_settings(
        self, *, metadata_item: MetadataRepositoryItem | None = None
    ) -> ProbePositionsRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.exception(''.join(exc.args))
            builder = self._builder_factory.create_default()

        return ProbePositionsRepositoryItem(
            self._rng, self._settings, builder, metadata_item=metadata_item
        )
