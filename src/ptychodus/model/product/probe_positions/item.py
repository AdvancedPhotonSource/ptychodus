from __future__ import annotations
import logging

import numpy

from ptychodus.api.affine import AffineTransform, transform_probe_positions
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ScanGeometry,
    calculate_scan_geometry,
)
from ptychodus.api.scattering.frame import resolve_exit_frame_transform
from ptychodus.api.scattering.geometry import ScatteringGeometry

from ...observer import Observable
from ...parameters import ParameterGroup
from ..metadata import MetadataRepositoryItem
from .builder import FromMemoryProbePositionsBuilder, ProbePositionsBuilder
from .settings import ProbePositionsSettings

logger = logging.getLogger(__name__)


class ProbePositionsRepositoryItem(ParameterGroup):
    """A product's probe positions, exposed in the exit frame of its scattering geometry.

    The builder yields positions in the lab frame, as the scan recorded them. They are
    carried into the exit frame of the product's geometry by
    :func:`ptychodus.api.scattering.frame.resolve_exit_frame_transform` and re-carried whenever the
    geometry changes, so editing an angle moves them exactly. Positions that arrive in the
    exit frame -- reconstruction output, a loaded product -- are taken back to the lab
    frame once, by the inverse, so they are never transformed twice. Without a metadata
    item the transform is the identity.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        builder: ProbePositionsBuilder,
        *,
        metadata_item: MetadataRepositoryItem | None = None,
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._builder = builder
        self._metadata_item = metadata_item
        self._probe_positions = ProbePositionSequence()
        self._geometry: ScanGeometry | None = None
        self._exit_frame_key: tuple[ScatteringGeometry, bool] | None = None
        self._exit_transform = AffineTransform.create_identity()
        self._resolve_exit_transform()

        if metadata_item is not None:
            metadata_item.add_observer(self)

        self._add_group('builder', builder, observe=True)

        self.expand_bbox = settings.expand_bbox.copy()
        self._add_parameter('expand_bbox', self.expand_bbox)

        self.expand_bbox_xmin_m = settings.expand_bbox_xmin_m.copy()
        self._add_parameter('expand_bbox_xmin_m', self.expand_bbox_xmin_m)

        self.expand_bbox_xmax_m = settings.expand_bbox_xmax_m.copy()
        self._add_parameter('expand_bbox_xmax_m', self.expand_bbox_xmax_m)

        self.expand_bbox_ymin_m = settings.expand_bbox_ymin_m.copy()
        self._add_parameter('expand_bbox_ymin_m', self.expand_bbox_ymin_m)

        self.expand_bbox_ymax_m = settings.expand_bbox_ymax_m.copy()
        self._add_parameter('expand_bbox_ymax_m', self.expand_bbox_ymax_m)

        self._rebuild()

    def assign_item(self, item: ProbePositionsRepositoryItem) -> None:
        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, scan: ProbePositionSequence) -> None:
        """Take positions already in the exit frame of the product's current geometry."""
        builder = FromMemoryProbePositionsBuilder(self._rng, self._settings, self._to_lab(scan))
        self.set_builder(builder)

    def get_exit_transform(self) -> AffineTransform:
        """The map from the builder's lab-frame positions to the exposed ones."""
        return self._exit_transform

    def _to_lab(self, scan: ProbePositionSequence) -> ProbePositionSequence:
        if self._exit_transform == AffineTransform.create_identity():
            return scan

        return ProbePositionSequence(
            list(transform_probe_positions(scan, self._exit_transform.inverted()))
        )

    def _resolve_exit_transform(self) -> bool:
        """Re-resolve the exit-frame transform; return whether it changed.

        Resolved only when the geometry or the propagation regime changes, so an
        unsupported geometry is warned about once rather than on every metadata edit.
        """
        if self._metadata_item is None:
            return False

        geometry = self._metadata_item.get_scattering_geometry()
        far_field = self._metadata_item.far_field.get_value()
        key = (geometry, far_field)

        if key == self._exit_frame_key:
            return False

        self._exit_frame_key = key
        transform = resolve_exit_frame_transform(geometry, far_field=far_field)
        is_changed = transform != self._exit_transform
        self._exit_transform = transform
        return is_changed

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_probe_positions(self) -> ProbePositionSequence:
        return self._probe_positions

    def get_builder(self) -> ProbePositionsBuilder:
        return self._builder

    def set_builder(self, builder: ProbePositionsBuilder) -> None:
        group = 'builder'
        self._remove_group(group)
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group(group, self._builder, observe=True)
        self._rebuild()

    def get_geometry(self) -> ScanGeometry | None:
        if self.expand_bbox.get_value():
            minimum_x_m = self.expand_bbox_xmin_m.get_value()
            maximum_x_m = self.expand_bbox_xmax_m.get_value()
            minimum_y_m = self.expand_bbox_ymin_m.get_value()
            maximum_y_m = self.expand_bbox_ymax_m.get_value()
            lenth_m = 0.0

            if self._geometry is not None:
                minimum_x_m = min(minimum_x_m, self._geometry.minimum_x_m)
                maximum_x_m = max(maximum_x_m, self._geometry.maximum_x_m)
                minimum_y_m = min(minimum_y_m, self._geometry.minimum_y_m)
                maximum_y_m = max(maximum_y_m, self._geometry.maximum_y_m)
                lenth_m = self._geometry.length_m

            return ScanGeometry(
                minimum_x_m=minimum_x_m,
                maximum_x_m=maximum_x_m,
                minimum_y_m=minimum_y_m,
                maximum_y_m=maximum_y_m,
                length_m=lenth_m,
            )

        return self._geometry

    def _rebuild(self) -> None:
        try:
            probe_positions = self._builder.build()
        except Exception:
            logger.exception('Failed to rebuild scan!')
            return

        if self._exit_transform != AffineTransform.create_identity():
            probe_positions = ProbePositionSequence(
                list(transform_probe_positions(probe_positions, self._exit_transform))
            )

        # build() always returns a ProbePositionSequence, and the class has no
        # mutators, so there is nothing to defend against by copying. This path
        # runs once per reconstructor iteration; the old round-trip through
        # Python dataclasses was O(N) every time.
        self._probe_positions = probe_positions
        self._geometry = calculate_scan_geometry(probe_positions)
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._metadata_item:
            if self._resolve_exit_transform():
                if isinstance(self._builder, FromMemoryProbePositionsBuilder):
                    # In-memory positions come with the object and probe they were solved or
                    # saved with, and those are not resampled into the new frame.
                    logger.warning(
                        'The scattering geometry changed; the positions follow it, but an '
                        'object and probe from a reconstruction or a product file stay in the '
                        'previous exit frame. Reconstruct again from new initial guesses.'
                    )

                self._rebuild()
        else:
            super()._update(observable)
