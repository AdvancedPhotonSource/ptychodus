import logging

import numpy.random

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider

from ...diffraction import AssembledDiffractionDataset
from .builder import FromMemoryProbeBuilder
from .builder_factory import ProbeBuilderFactory
from .item import ProbeRepositoryItem
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

    def _warn_if_conditioning_ignored(self) -> None:
        """Note that the mode settings do not apply to in-memory probes.

        A probe supplied in memory comes from reconstruction output or a product
        loaded from file, so its mode structure is already what the reconstructor
        solved for. Batch mode reads product.h5 through this path, where a user
        who sets a mode count in settings.ini would otherwise see it silently do
        nothing. Set the mode counts on the run that produces the probe instead --
        ptychodus-bdp reads probes through the from-file builder, which does
        condition them.
        """
        settings = self._settings
        # The decay parameters and the orthogonalization flag are inert at a
        # single incoherent mode, so gate on the two counts alone; including them
        # would fire on default settings.
        is_conditioning_requested = (
            settings.num_incoherent_modes.get_value() != 1
            or settings.num_coherent_modes.get_value() != 1
        )

        if is_conditioning_requested:
            logger.info(
                'Probes supplied in memory are already conditioned;'
                ' ignoring the incoherent and coherent mode settings.'
            )

    def create(
        self, geometry_provider: ProbeGeometryProvider, probe: ProbeSequence | None = None
    ) -> ProbeRepositoryItem:
        if probe is None:
            builder = self._builder_factory.create_default()
        else:
            self._warn_if_conditioning_ignored()
            builder = FromMemoryProbeBuilder(self._rng, self._settings, probe)

        return ProbeRepositoryItem(self._rng, geometry_provider, self._settings, builder)

    def create_from_settings(
        self,
        geometry_provider: ProbeGeometryProvider,
        *,
        dataset: AssembledDiffractionDataset | None = None,
    ) -> ProbeRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings(dataset=dataset)
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builder_factory.create_default()

        return ProbeRepositoryItem(self._rng, geometry_provider, self._settings, builder)
