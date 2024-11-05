from collections.abc import Iterator, Sequence


class PtyChiEnumerators:
    def __init__(self) -> None:
        try:
            from ptychi.api import LossFunctions, Optimizers, OrthogonalizationMethods
        except ModuleNotFoundError:
            self._lossFunctions: Sequence[str] = list()
            self._optimizers: Sequence[str] = list()
            self._orthogonalizationMethods: Sequence[str] = list()
        else:
            self._lossFunctions = [member.name for member in LossFunctions]
            self._optimizers = [member.name for member in Optimizers]
            self._orthogonalizationMethods = [member.name for member in OrthogonalizationMethods]

    def lossFunctions(self) -> Iterator[str]:
        return iter(self._lossFunctions)

    def optimizers(self) -> Iterator[str]:
        return iter(self._optimizers)

    def orthogonalizationMethods(self) -> Iterator[str]:
        return iter(self._orthogonalizationMethods)
