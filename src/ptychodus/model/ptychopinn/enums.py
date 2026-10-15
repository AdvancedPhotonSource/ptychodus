from collections.abc import Iterator, Sequence


class PtychoPINNEnumerators:
    """Allowed values for the string-valued PtychoPINN settings.

    Each list mirrors the corresponding ``Literal`` in the backend's
    configuration dataclasses, so a value chosen here resolves rather than
    failing validation once it reaches the training payload.
    """

    def __init__(self) -> None:
        self._data_normalization_modes: Sequence[str] = ['Group', 'Batch', 'None']
        self._data_scaling_methods: Sequence[str] = ['Parseval', 'Max']

        self._loss_functions: Sequence[str] = ['Poisson', 'MAE']
        self._amplitude_activation_functions: Sequence[str] = ['silu', 'sigmoid']
        self._auxiliary_loss_functions: Sequence[str] = [
            'None',
            'Total_Variation',
            'Mean_Deviation',
        ]

        self._learning_rate_schedulers: Sequence[str] = [
            'Default',
            'Exponential',
            'MultiStage',
            'Adaptive',
            'WarmupCosine',
            'ReduceLROnPlateau',
        ]
        self._physics_weight_schedules: Sequence[str] = ['linear', 'cosine', 'exponential']

        self._patch_weighting_methods: Sequence[str] = ['probe', 'uniform']

        self._accelerators: Sequence[str] = ['auto', 'cpu', 'gpu', 'cuda', 'mps']

    def get_data_normalization_modes(self) -> Iterator[str]:
        return iter(self._data_normalization_modes)

    def get_data_scaling_methods(self) -> Iterator[str]:
        return iter(self._data_scaling_methods)

    def get_loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def get_amplitude_activation_functions(self) -> Iterator[str]:
        return iter(self._amplitude_activation_functions)

    def get_auxiliary_loss_functions(self) -> Iterator[str]:
        return iter(self._auxiliary_loss_functions)

    def get_learning_rate_schedulers(self) -> Iterator[str]:
        return iter(self._learning_rate_schedulers)

    def get_physics_weight_schedules(self) -> Iterator[str]:
        return iter(self._physics_weight_schedules)

    def get_patch_weighting_methods(self) -> Iterator[str]:
        return iter(self._patch_weighting_methods)

    def get_accelerators(self) -> Iterator[str]:
        return iter(self._accelerators)
