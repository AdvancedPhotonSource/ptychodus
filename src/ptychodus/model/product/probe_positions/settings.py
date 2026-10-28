from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parameters import Parameter
from ptychodus.api.settings import SettingsRegistry

from ...declared import ParameterDescription, declare_in


class ProbePositionsSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('ProbePositions')
        self._group.add_observer(self)

        self.builder = self._group.create_string_parameter('Builder', 'rectangular_raster')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/scan.csv'))
        self.file_type = self._group.create_string_parameter('FileType', 'CSV')

        self.affine00 = self._group.create_real_parameter('Affine00', 1.0)
        self.affine01 = self._group.create_real_parameter('Affine01', 0.0)
        self.affine02 = self._group.create_real_parameter('Affine02', 0.0)
        self.affine10 = self._group.create_real_parameter('Affine10', 0.0)
        self.affine11 = self._group.create_real_parameter('Affine11', 1.0)
        self.affine12 = self._group.create_real_parameter('Affine12', 0.0)
        self.jitter_radius_m = self._group.create_real_parameter(
            'JitterRadiusInMeters', 0.0, minimum=0.0
        )
        self.num_discard_at_start = self._group.create_integer_parameter(
            'NumberOfPointsToDiscardAtStart', 0, minimum=0
        )
        self.num_discard_at_end = self._group.create_integer_parameter(
            'NumberOfPointsToDiscardAtEnd', 0, minimum=0
        )

        self.expand_bbox = self._group.create_boolean_parameter('ExpandBoundingBox', False)
        self.expand_bbox_xmin_m = self._group.create_real_parameter(
            'ExpandedBoundingBoxMinimumXInMeters', -5e-7
        )
        self.expand_bbox_xmax_m = self._group.create_real_parameter(
            'ExpandedBoundingBoxMaximumXInMeters', +5e-7
        )
        self.expand_bbox_ymin_m = self._group.create_real_parameter(
            'ExpandedBoundingBoxMinimumYInMeters', -5e-7
        )
        self.expand_bbox_ymax_m = self._group.create_real_parameter(
            'ExpandedBoundingBoxMaximumYInMeters', +5e-7
        )

    def declare(self, described: Sequence[ParameterDescription]) -> Mapping[str, Parameter[Any]]:
        """Create the parameters a scan pattern declares, and return them by key.

        The pattern-specific keys of this section are not written out above,
        because which of them exist is settled by which generators registered.
        They land in the same group all the same, so one section still holds
        everything a scan is configured by.
        """
        return declare_in(self._group, described)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
