from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ScanSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Scan')
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

        self.num_points_x = self._group.create_integer_parameter('NumberOfPointsX', 10, minimum=0)
        self.num_points_y = self._group.create_integer_parameter('NumberOfPointsY', 10, minimum=0)
        self.step_size_x_m = self._group.create_real_parameter(
            'StepSizeXInMeters', 1e-6, minimum=0.0
        )
        self.step_size_y_m = self._group.create_real_parameter(
            'StepSizeYInMeters', 1e-6, minimum=0.0
        )

        self.radial_step_size_m = self._group.create_real_parameter(
            'RadialStepSizeInMeters', 1e-6, minimum=0.0
        )
        self.num_shells = self._group.create_integer_parameter('NumberOfShells', 5, minimum=0)
        self.num_points_in_first_shell = self._group.create_integer_parameter(
            'NumberOfPointsInFirstShell', 10, minimum=0
        )

        self.amplitude_x_m = self._group.create_real_parameter(
            'AmplitudeXInMeters', 4.5e-6, minimum=0.0
        )
        self.amplitude_y_m = self._group.create_real_parameter(
            'AmplitudeYInMeters', 4.5e-6, minimum=0.0
        )
        self.angular_step_x_turns = self._group.create_real_parameter('AngularStepXInTurns', 0.03)
        self.angular_step_y_turns = self._group.create_real_parameter('AngularStepYInTurns', 0.04)
        self.angular_shift_turns = self._group.create_real_parameter('AngularShiftInTurns', 0.25)

        self.radius_scalar_m = self._group.create_real_parameter(
            'RadiusScalarInMeters', 0.5e-6, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
