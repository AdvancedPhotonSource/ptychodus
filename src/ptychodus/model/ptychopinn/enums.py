from collections.abc import Iterator, Sequence


class PtychoPINNEnumerators:
    def __init__(self) -> None:
        self._amp_activations: Sequence[str] = ['sigmoid', 'swish', 'softplus', 'relu']

    def get_amp_activations(self) -> Iterator[str]:
        return iter(self._amp_activations)
