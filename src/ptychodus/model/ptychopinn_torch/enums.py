from collections.abc import Iterator, Sequence


class PtychoPINNTorchEnumerators:
    def __init__(self) -> None:
        self._data_normalization_modes: Sequence[str] = ['Group', 'Batch']
        self._neighbor_lookup_methods: Sequence[str] = ['Nearest', 'Min_dist', '4_quadrant']
        self._scan_patterns: Sequence[str] = ['Isotropic', 'Rectangular']
        self._data_scaling_methods: Sequence[str] = ['Parseval', 'Max']

        self._loss_functions: Sequence[str] = ['Poisson', 'MAE']
        self._amplitude_activation_functions: Sequence[str] = ['silu', 'sigmoid']
        self._auxiliary_loss_functions: Sequence[str] = [
            'None',
            'Total_Variation',
            'Mean_Deviation',
        ]

        self._devices: Sequence[str] = ['cuda', 'cpu']
        self._learning_rate_schedulers: Sequence[str] = [
            'Default',
            'Exponential',
            'MultiStage',
            'Adaptive',
            'Cosine',
        ]
        self._physics_weight_schedules: Sequence[str] = ['linear', 'cosine', 'exponential']
        self._torch_loss_modes: Sequence[str] = ['poisson', 'mae']

        self._patch_weighting_methods: Sequence[str] = ['probe', 'uniform']

    def get_data_normalization_modes(self) -> Iterator[str]:
        return iter(self._data_normalization_modes)

    def get_neighbor_lookup_methods(self) -> Iterator[str]:
        return iter(self._neighbor_lookup_methods)

    def get_scan_patterns(self) -> Iterator[str]:
        return iter(self._scan_patterns)

    def get_data_scaling_methods(self) -> Iterator[str]:
        return iter(self._data_scaling_methods)

    def get_loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def get_amplitude_activation_functions(self) -> Iterator[str]:
        return iter(self._amplitude_activation_functions)

    def get_auxiliary_loss_functions(self) -> Iterator[str]:
        return iter(self._auxiliary_loss_functions)

    def get_devices(self) -> Iterator[str]:
        return iter(self._devices)

    def get_learning_rate_schedulers(self) -> Iterator[str]:
        return iter(self._learning_rate_schedulers)

    def get_physics_weight_schedules(self) -> Iterator[str]:
        return iter(self._physics_weight_schedules)

    def get_torch_loss_modes(self) -> Iterator[str]:
        return iter(self._torch_loss_modes)

    def get_patch_weighting_methods(self) -> Iterator[str]:
        return iter(self._patch_weighting_methods)
