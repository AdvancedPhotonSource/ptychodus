from collections.abc import Iterator, Sequence


class PtychoFMEnumerators:
    def __init__(self) -> None:
        self._sharding_strategies: Sequence[str] = ['static', 'dynamic']
        self._encoder_types: Sequence[str] = ['custom', 'pretrained']
        self._init_methods: Sequence[str] = ['trunc_normal', 'kaiming']
        self._loss_functions: Sequence[str] = [
            'smooth_l1',
            'mse',
            'l1',
            'poisson_nll',
            'weighted',
        ]
        self._weighted_loss_types: Sequence[str] = ['mse', 'mae']
        self._training_modes: Sequence[str] = ['unsupervised', 'supervised']

    def get_sharding_strategies(self) -> Iterator[str]:
        return iter(self._sharding_strategies)

    def get_encoder_types(self) -> Iterator[str]:
        return iter(self._encoder_types)

    def get_init_methods(self) -> Iterator[str]:
        return iter(self._init_methods)

    def get_loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def get_weighted_loss_types(self) -> Iterator[str]:
        return iter(self._weighted_loss_types)

    def get_training_modes(self) -> Iterator[str]:
        return iter(self._training_modes)
