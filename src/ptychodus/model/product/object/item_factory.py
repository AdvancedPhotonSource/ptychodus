import logging

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from ...diffraction import AssembledDiffractionDataset
from .builder import FromMemoryObjectBuilder
from .builder_factory import ObjectBuilderFactory
from .item import ObjectRepositoryItem
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        builder_factory: ObjectBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builder_factory = builder_factory

    def _warn_if_conditioning_ignored(self) -> None:
        """Note that conditioning settings do not apply to in-memory objects.

        An object supplied in memory comes from reconstruction output or a product
        loaded from file, so its layer structure and canvas size are already what
        the reconstructor solved for. Batch mode reads product.h5 through this
        path, where a user who sets a layer spacing in settings.ini would
        otherwise see it silently do nothing. Set it on the run that produces the
        object instead -- the from-file builder does apply the layer spacing.
        """
        settings = self._settings
        is_conditioning_requested = (
            # Note the padding parameters default to 1, not 0.
            settings.extra_padding_x.get_value() != 1
            or settings.extra_padding_y.get_value() != 1
            or len(settings.object_layer_spacing_m.get_value()) != 0
        )

        if is_conditioning_requested:
            logger.info(
                'Objects supplied in memory are already conditioned;'
                ' ignoring the extra padding and layer spacing settings.'
            )

    def create(
        self, geometry_provider: ObjectGeometryProvider, object_: Object | None = None
    ) -> ObjectRepositoryItem:
        if object_ is None:
            builder = self._builder_factory.create_default()
        else:
            self._warn_if_conditioning_ignored()
            builder = FromMemoryObjectBuilder(self._settings, object_)

        return ObjectRepositoryItem(geometry_provider, self._settings, builder)

    def create_from_settings(
        self,
        geometry_provider: ObjectGeometryProvider,
        *,
        dataset: AssembledDiffractionDataset | None = None,
    ) -> ObjectRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings(dataset=dataset)
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builder_factory.create_default()

        return ObjectRepositoryItem(geometry_provider, self._settings, builder)
