from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Object')
        self._group.add_observer(self)

        self.builder = self._group.create_string_parameter('Builder', 'Random')
        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/object.npy'))
        self.file_type = self._group.create_string_parameter('FileType', 'NPY')

        self.object_layer_spacing_m = self._group.create_real_sequence_parameter(
            'ObjectLayerSpacingInMeters', []
        )

        self.extra_padding_x = self._group.create_integer_parameter('ExtraPaddingX', 1, minimum=0)
        self.extra_padding_y = self._group.create_integer_parameter('ExtraPaddingY', 1, minimum=0)
        self.amplitude_mean = self._group.create_real_parameter(
            'AmplitudeMean', 1.0, minimum=0.0, maximum=1.0
        )
        self.amplitude_deviation = self._group.create_real_parameter(
            'AmplitudeDeviation', 0.0, minimum=0.0, maximum=1.0
        )
        self.phase_deviation_tr = self._group.create_real_parameter(
            'PhaseDeviationInTurns', 0.0, minimum=0.0, maximum=1.0
        )
        self.blur_deviation_px = self._group.create_real_parameter(
            'BlurDeviation', 0.0, minimum=0.0
        )

        self.leaf_radius_lower_px = self._group.create_real_parameter(
            'LeafRadiusLowerInPixels', 2.0, minimum=0.0
        )
        self.leaf_radius_upper_px = self._group.create_real_parameter(
            'LeafRadiusUpperInPixels', 999.0, minimum=0.0
        )
        self.leaf_radius_power_law_exponent = self._group.create_real_parameter(
            'LeafRadiusPowerLawExponent', 3.0, minimum=1.0
        )
        self.leaf_amplitude_lower = self._group.create_real_parameter(
            'LeafAmplitudeLower', 0.0, minimum=0.0, maximum=1.0
        )
        self.leaf_amplitude_upper = self._group.create_real_parameter(
            'LeafAmplitudeUpper', 1.0, minimum=0.0, maximum=1.0
        )
        self.leaf_phase_lower_tr = self._group.create_real_parameter(
            'LeafPhaseLowerInTurns',
            -0.5,
        )
        self.leaf_phase_upper_tr = self._group.create_real_parameter(
            'LeafPhaseUpperInTurns',
            0.5,
        )

        self.correlation_length_px = self._group.create_real_parameter(
            'CorrelationLengthInPixels', 30.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
