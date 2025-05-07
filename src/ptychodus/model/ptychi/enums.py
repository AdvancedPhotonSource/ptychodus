from collections.abc import Iterator, Sequence


class PtyChiEnumerators:
    def __init__(self) -> None:
        try:
            from ptychi.api import (
                BatchingModes,
                Directions,
                ForwardModels,
                ImageGradientMethods,
                ImageIntegrationMethods,
                LossFunctions,
                NoiseModels,
                OPRWeightSmoothingMethods,
                Optimizers,
                OrthogonalizationMethods,
                PatchInterpolationMethods,
                PositionCorrectionTypes,
            )
        except ModuleNotFoundError:
            self._batching_modes: Sequence[str] = list()
            self._directions: Sequence[str] = list()
            self._forward_models: Sequence[str] = list()
            self._image_gradient_methods: Sequence[str] = list()
            self._image_integration_methods: Sequence[str] = list()
            self._loss_functions: Sequence[str] = list()
            self._noise_models: Sequence[str] = list()
            self._opr_weight_smoothing_methods: Sequence[str] = list()
            self._optimizers: Sequence[str] = list()
            self._orthogonalization_methods: Sequence[str] = list()
            self._patch_interpolation_methods: Sequence[str] = list()
            self._position_correction_types: Sequence[str] = list()
        else:
            self._batching_modes = [member.name for member in BatchingModes]
            self._directions = [member.name for member in Directions]
            self._forward_models = [member.name for member in ForwardModels]
            self._image_gradient_methods = [member.name for member in ImageGradientMethods]
            self._image_integration_methods = [member.name for member in ImageIntegrationMethods]
            self._loss_functions = [member.name for member in LossFunctions]
            self._noise_models = [member.name for member in NoiseModels]
            self._opr_weight_smoothing_methods = [
                member.name for member in OPRWeightSmoothingMethods
            ]
            self._optimizers = [member.name for member in Optimizers]
            self._orthogonalization_methods = [member.name for member in OrthogonalizationMethods]
            self._patch_interpolation_methods = [
                member.name for member in PatchInterpolationMethods
            ]
            self._position_correction_types = [member.name for member in PositionCorrectionTypes]

    def batching_modes(self) -> Iterator[str]:
        return iter(self._batching_modes)

    def directions(self) -> Iterator[str]:
        return iter(self._directions)

    def forward_models(self) -> Iterator[str]:
        return iter(self._forward_models)

    def image_gradient_methods(self) -> Iterator[str]:
        return iter(self._image_gradient_methods)

    def image_integration_methods(self) -> Iterator[str]:
        return iter(self._image_integration_methods)

    def loss_functions(self) -> Iterator[str]:
        return iter(self._loss_functions)

    def noise_models(self) -> Iterator[str]:
        return iter(self._noise_models)

    def opr_weight_smoothing_methods(self) -> Iterator[str]:
        return iter(self._opr_weight_smoothing_methods)

    def optimizers(self) -> Iterator[str]:
        return iter(self._optimizers)

    def orthogonalization_methods(self) -> Iterator[str]:
        return iter(self._orthogonalization_methods)

    def patch_interpolation_methods(self) -> Iterator[str]:
        return iter(self._patch_interpolation_methods)

    def position_correction_types(self) -> Iterator[str]:
        return iter(self._position_correction_types)
