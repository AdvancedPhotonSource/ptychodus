from collections.abc import Iterator, Sequence


class PtyChiEnumerators:
    def __init__(self) -> None:
        try:
            from ptychi.api import (
                BatchingModes,
                Directions,
                ImageGradientMethods,
                ImageIntegrationMethods,
                LossFunctions,
                OPRWeightSmoothingMethods,
                Optimizers,
                OrthogonalizationMethods,
                PatchInterpolationMethods,
                PositionCorrectionTypes,
            )
        except ModuleNotFoundError:
            self._batchingModes: Sequence[str] = list()
            self._directions: Sequence[str] = list()
            self._imageGradientMethods: Sequence[str] = list()
            self._imageIntegrationMethods: Sequence[str] = list()
            self._lossFunctions: Sequence[str] = list()
            self._oprWeightSmoothingMethods: Sequence[str] = list()
            self._optimizers: Sequence[str] = list()
            self._orthogonalizationMethods: Sequence[str] = list()
            self._patchInterpolationMethods: Sequence[str] = list()
            self._positionCorrectionTypes: Sequence[str] = list()
        else:
            self._batchingModes = [member.name for member in BatchingModes]
            self._directions = [member.name for member in Directions]
            self._imageGradientMethods = [member.name for member in ImageGradientMethods]
            self._imageIntegrationMethods = [member.name for member in ImageIntegrationMethods]
            self._lossFunctions = [member.name for member in LossFunctions]
            self._oprWeightSmoothingMethods = [member.name for member in OPRWeightSmoothingMethods]
            self._optimizers = [member.name for member in Optimizers]
            self._orthogonalizationMethods = [member.name for member in OrthogonalizationMethods]
            self._patchInterpolationMethods = [member.name for member in PatchInterpolationMethods]
            self._positionCorrectionTypes = [member.name for member in PositionCorrectionTypes]

    def batchingModes(self) -> Iterator[str]:
        return iter(self._batchingModes)

    def directions(self) -> Iterator[str]:
        return iter(self._directions)

    def imageGradientMethods(self) -> Iterator[str]:
        return iter(self._imageGradientMethods)

    def imageIntegrationMethods(self) -> Iterator[str]:
        return iter(self._imageIntegrationMethods)

    def lossFunctions(self) -> Iterator[str]:
        return iter(self._lossFunctions)

    def oprWeightSmoothingMethods(self) -> Iterator[str]:
        return iter(self._oprWeightSmoothingMethods)

    def optimizers(self) -> Iterator[str]:
        return iter(self._optimizers)

    def orthogonalizationMethods(self) -> Iterator[str]:
        return iter(self._orthogonalizationMethods)

    def patchInterpolationMethods(self) -> Iterator[str]:
        return iter(self._patchInterpolationMethods)

    def positionCorrectionTypes(self) -> Iterator[str]:
        return iter(self._positionCorrectionTypes)
