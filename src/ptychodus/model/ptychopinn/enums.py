from collections.abc import Iterator, Sequence


class PtychoPINNEnumerators:
    def __init__(self) -> None:
        self._amp_activations: Sequence[str] = ['sigmoid', 'swish', 'softplus', 'relu']
        self._data_sources: Sequence[str] = [
            'V',
            'diagonals',
            'dset',
            'experimental',
            'generic',
            'grf',
            'lines',
            'points',
            'testimg',
            'xpp',
        ]
        self._model_types: Sequence[str] = ['pinn', 'supervised']

    def get_amp_activations(self) -> Iterator[str]:
        return iter(self._amp_activations)

    def get_data_sources(self) -> Iterator[str]:
        return iter(self._data_sources)

    def get_model_types(self) -> Iterator[str]:
        return iter(self._model_types)
