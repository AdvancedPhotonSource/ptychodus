from collections.abc import Iterator, Sequence


class PtyChiEnumerators:
    def __init__(self) -> None:
        try:
            from ptychi.api import (
                BatchingModes,
                Directions,
                ImageGradientMethods,
                LossFunctions,
                Optimizers,
                OrthogonalizationMethods,
                PositionCorrectionTypes,
            )
        except ModuleNotFoundError:
            self._batchingModes: Sequence[str] = list()
            self._directions: Sequence[str] = list()
            self._imageGradientMethods: Sequence[str] = list()
            self._lossFunctions: Sequence[str] = list()
            self._optimizers: Sequence[str] = list()
            self._orthogonalizationMethods: Sequence[str] = list()
            self._positionCorrectionTypes: Sequence[str] = list()
        else:
            self._batchingModes = [member.name for member in BatchingModes]
            self._directions = [member.name for member in Directions]
            self._imageGradientMethods = [member.name for member in ImageGradientMethods]
            self._lossFunctions = [member.name for member in LossFunctions]
            self._optimizers = [member.name for member in Optimizers]
            self._orthogonalizationMethods = [member.name for member in OrthogonalizationMethods]
            self._positionCorrectionTypes = [member.name for member in PositionCorrectionTypes]

    def optimizers(self) -> Iterator[str]:
        return iter(self._optimizers)

    def directions(self) -> Iterator[str]:
        return iter(self._directions)

    def imageGradientMethods(self) -> Iterator[str]:
        return iter(self._imageGradientMethods)

    def orthogonalizationMethods(self) -> Iterator[str]:
        return iter(self._orthogonalizationMethods)

    def positionCorrectionTypes(self) -> Iterator[str]:
        return iter(self._positionCorrectionTypes)

    def batchingModes(self) -> Iterator[str]:
        return iter(self._batchingModes)

    def lossFunctions(self) -> Iterator[str]:
        return iter(self._lossFunctions)
