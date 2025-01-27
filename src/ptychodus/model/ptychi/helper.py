import logging

import numpy

from ptychi.api import (
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
    ImageGradientMethods,
    ImageIntegrationMethods,
    LossFunctions,
    NoiseModels,
    OPRWeightSmoothingMethods,
    OptimizationPlan,
    OptimizationPlan,
    Optimizers,
    Optimizers,
    OrthogonalizationMethods,
    PatchInterpolationMethods,
    PositionCorrectionTypes,
    PtychographyDataOptions,
)
from ptychi.api.options.base import (
    ObjectL1NormConstraintOptions,
    ObjectMultisliceRegularizationOptions,
    ObjectSmoothnessConstraintOptions,
    ObjectTotalVariationOptions,
    OPRModeWeightsSmoothingOptions,
    PositionCorrectionOptions,
    ProbeCenterConstraintOptions,
    ProbeOrthogonalizeIncoherentModesOptions,
    ProbeOrthogonalizeOPRModesOptions,
    ProbePositionMagnitudeLimitOptions,
    ProbePowerConstraintOptions,
    ProbeSupportConstraintOptions,
    RemoveGridArtifactsOptions,
)

from ptychodus.api.object import Object
from ptychodus.api.probe import Probe
from ptychodus.api.product import ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.typing import ComplexArrayType, RealArrayType

from ..patterns import Detector
from .settings import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)


__all__ = ['PtyChiOptionsHelper']

logger = logging.getLogger(__name__)


def create_optimization_plan(start: int, stop: int, stride: int) -> OptimizationPlan:
    return OptimizationPlan(start, None if stop < 0 else stop, stride)


def parse_optimizer(text: str) -> Optimizers:
    try:
        optimizer = Optimizers[text.upper()]
    except KeyError:
        logger.warning('Failed to parse optimizer "{text}"!')
        optimizer = Optimizers.SGD

    return optimizer


class PtyChiReconstructorOptionsHelper:
    def __init__(self, settings: PtyChiReconstructorSettings) -> None:
        self._settings = settings

    @property
    def num_epochs(self) -> int:
        return self._settings.numEpochs.getValue()

    @property
    def batch_size(self) -> int:
        return self._settings.batchSize.getValue()

    @property
    def batching_mode(self) -> BatchingModes:
        batching_mode_str = self._settings.batchingMode.getValue()

        try:
            return BatchingModes[batching_mode_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{batching_mode_str}"!')
            return BatchingModes.RANDOM

    @property
    def compact_mode_update_clustering(self) -> bool:
        return self._settings.batchStride.getValue() > 0

    @property
    def compact_mode_update_clustering_stride(self) -> int:
        return self._settings.batchStride.getValue()

    @property
    def default_device(self) -> Devices:
        return Devices.GPU if self._settings.useDevices.getValue() else Devices.CPU

    @property
    def default_dtype(self) -> Dtypes:
        return Dtypes.FLOAT64 if self._settings.useDoublePrecision.getValue() else Dtypes.FLOAT32

    @property
    def random_seed(self) -> int | None:
        return None  # TODO

    @property
    def displayed_loss_function(self) -> LossFunctions | None:
        return LossFunctions.MSE_SQRT  # TODO

    @property
    def use_low_memory_forward_model(self) -> bool:
        return self._settings.useLowMemoryForwardModel.getValue()


class PtyChiObjectOptionsHelper:
    def __init__(self, settings: PtyChiObjectSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.isOptimizable.getValue()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimizationPlanStart.getValue(),
            self._settings.optimizationPlanStop.getValue(),
            self._settings.optimizationPlanStride.getValue(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.getValue())

    @property
    def step_size(self) -> float:
        return self._settings.stepSize.getValue()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def l1_norm_constraint(self) -> ObjectL1NormConstraintOptions:
        return ObjectL1NormConstraintOptions(
            enabled=self._settings.constrainL1Norm.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainL1NormStart.getValue(),
                self._settings.constrainL1NormStop.getValue(),
                self._settings.constrainL1NormStride.getValue(),
            ),
            weight=self._settings.constrainL1NormWeight.getValue(),
        )

    @property
    def smoothness_constraint(self) -> ObjectSmoothnessConstraintOptions:
        return ObjectSmoothnessConstraintOptions(
            enabled=self._settings.constrainSmoothness.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainSmoothnessStart.getValue(),
                self._settings.constrainSmoothnessStop.getValue(),
                self._settings.constrainSmoothnessStride.getValue(),
            ),
            alpha=self._settings.constrainSmoothnessAlpha.getValue(),
        )

    @property
    def total_variation(self) -> ObjectTotalVariationOptions:
        return ObjectTotalVariationOptions(
            enabled=self._settings.constrainTotalVariation.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainTotalVariationStart.getValue(),
                self._settings.constrainTotalVariationStop.getValue(),
                self._settings.constrainTotalVariationStride.getValue(),
            ),
            weight=self._settings.constrainTotalVariationWeight.getValue(),
        )

    @property
    def remove_grid_artifacts(self) -> RemoveGridArtifactsOptions:
        direction_str = self._settings.removeGridArtifactsDirection.getValue()

        try:
            direction = Directions[direction_str.upper()]
        except KeyError:
            logger.warning('Failed to parse direction "{direction_str}"!')
            direction = Directions.XY

        return RemoveGridArtifactsOptions(
            enabled=self._settings.removeGridArtifacts.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.removeGridArtifactsStart.getValue(),
                self._settings.removeGridArtifactsStop.getValue(),
                self._settings.removeGridArtifactsStride.getValue(),
            ),
            period_x_m=self._settings.removeGridArtifactsPeriodXInMeters.getValue(),
            period_y_m=self._settings.removeGridArtifactsPeriodYInMeters.getValue(),
            window_size=self._settings.removeGridArtifactsWindowSizeInPixels.getValue(),
            direction=direction,
        )

    @property
    def multislice_regularization(self) -> ObjectMultisliceRegularizationOptions:
        unwrap_image_grad_method_str = (
            self._settings.regularizeMultisliceUnwrapPhaseImageGradientMethod.getValue()
        )

        try:
            unwrap_image_grad_method = ImageGradientMethods[unwrap_image_grad_method_str.upper()]
        except KeyError:
            logger.warning(
                'Failed to parse image gradient method "{unwrap_image_grad_method_str}"!'
            )
            unwrap_image_grad_method = ImageGradientMethods.FOURIER_SHIFT

        unwrap_image_integration_method_str = (
            self._settings.regularizeMultisliceUnwrapPhaseImageIntegrationMethod.getValue()
        )

        try:
            unwrap_image_integration_method = ImageIntegrationMethods[
                unwrap_image_integration_method_str.upper()
            ]
        except KeyError:
            logger.warning(
                'Failed to parse image integrationient method "{unwrap_image_integration_method_str}"!'
            )
            unwrap_image_integration_method = ImageIntegrationMethods.DECONVOLUTION

        return ObjectMultisliceRegularizationOptions(
            enabled=self._settings.regularizeMultislice.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.regularizeMultisliceStart.getValue(),
                self._settings.regularizeMultisliceStop.getValue(),
                self._settings.regularizeMultisliceStride.getValue(),
            ),
            weight=self._settings.regularizeMultisliceWeight.getValue(),
            unwrap_phase=self._settings.regularizeMultisliceUnwrapPhase.getValue(),
            unwrap_image_grad_method=unwrap_image_grad_method,
            unwrap_image_integration_method=unwrap_image_integration_method,
        )

    @property
    def patch_interpolation_method(self) -> PatchInterpolationMethods:
        method_str = self._settings.patchInterpolator.getValue()

        try:
            return PatchInterpolationMethods[method_str.upper()]
        except KeyError:
            logger.warning('Failed to parse patch interpolation method "{method_str}"!')
            return PatchInterpolationMethods.FOURIER

    def get_initial_guess(self, object_: Object) -> ComplexArrayType:
        return object_.getArray()

    def get_slice_spacings_m(self, object_: Object) -> RealArrayType:
        return numpy.array(object_.layerDistanceInMeters[:-1])  # FIXME iff multislice

    def get_pixel_size_m(self, object_: Object) -> float:
        pixel_geometry = object_.getPixelGeometry()

        if pixel_geometry is None:
            logger.error('Missing object pixel geometry!')
            return 1.0

        return pixel_geometry.widthInMeters


class PtyChiProbeOptionsHelper:
    def __init__(self, settings: PtyChiProbeSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.isOptimizable.getValue()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimizationPlanStart.getValue(),
            self._settings.optimizationPlanStop.getValue(),
            self._settings.optimizationPlanStride.getValue(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.getValue())

    @property
    def step_size(self) -> float:
        return self._settings.stepSize.getValue()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def orthogonalize_incoherent_modes(self) -> ProbeOrthogonalizeIncoherentModesOptions:
        method_str = self._settings.orthogonalizeIncoherentModesMethod.getValue()

        try:
            method = OrthogonalizationMethods[method_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{method_str}"!')
            method = OrthogonalizationMethods.GS

        return ProbeOrthogonalizeIncoherentModesOptions(
            enabled=self._settings.orthogonalizeIncoherentModes.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.orthogonalizeIncoherentModesStart.getValue(),
                self._settings.orthogonalizeIncoherentModesStop.getValue(),
                self._settings.orthogonalizeIncoherentModesStride.getValue(),
            ),
            method=method,
        )

    @property
    def orthogonalize_opr_modes(self) -> ProbeOrthogonalizeOPRModesOptions:
        return ProbeOrthogonalizeOPRModesOptions(
            enabled=self._settings.orthogonalizeOPRModes.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.orthogonalizeOPRModesStart.getValue(),
                self._settings.orthogonalizeOPRModesStop.getValue(),
                self._settings.orthogonalizeOPRModesStride.getValue(),
            ),
        )

    @property
    def support_constraint(self) -> ProbeSupportConstraintOptions:
        return ProbeSupportConstraintOptions(
            enabled=self._settings.constrainSupport.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainSupportStart.getValue(),
                self._settings.constrainSupportStop.getValue(),
                self._settings.constrainSupportStride.getValue(),
            ),
            threshold=self._settings.constrainSupportThreshold.getValue(),
        )

    @property
    def center_constraint(self) -> ProbeCenterConstraintOptions:
        return ProbeCenterConstraintOptions(
            enabled=self._settings.constrainCenter.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainCenterStart.getValue(),
                self._settings.constrainCenterStop.getValue(),
                self._settings.constrainCenterStride.getValue(),
            ),
        )

    @property
    def eigenmode_update_relaxation(self) -> float:
        return self._settings.relaxEigenmodeUpdate.getValue()

    def get_initial_guess(self, probe: Probe) -> ComplexArrayType:
        return probe.getArray()

    def get_power_constraint(self, metadata: ProductMetadata) -> ProbePowerConstraintOptions:
        return ProbePowerConstraintOptions(
            enabled=self._settings.constrainProbePower.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.constrainProbePowerStart.getValue(),
                self._settings.constrainProbePowerStop.getValue(),
                self._settings.constrainProbePowerStride.getValue(),
            ),
            probe_power=metadata.probePhotonCount,
        )


class PtyChiProbePositionOptionsHelper:
    def __init__(self, settings: PtyChiProbePositionSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.isOptimizable.getValue()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimizationPlanStart.getValue(),
            self._settings.optimizationPlanStop.getValue(),
            self._settings.optimizationPlanStride.getValue(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.getValue())

    @property
    def step_size(self) -> float:
        return self._settings.stepSize.getValue()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()


class PtyChiOPROptionsHelper:
    def __init__(self, settings: PtyChiOPRSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.isOptimizable.getValue()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimizationPlanStart.getValue(),
            self._settings.optimizationPlanStop.getValue(),
            self._settings.optimizationPlanStride.getValue(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.getValue())

    @property
    def step_size(self) -> float:
        return self._settings.stepSize.getValue()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()


class PtyChiOptionsHelper:
    def __init__(
        self,
        reconstructor_settings: PtyChiReconstructorSettings,
        object_settings: PtyChiObjectSettings,
        probe_settings: PtyChiProbeSettings,
        probe_position_settings: PtyChiProbePositionSettings,
        opr_settings: PtyChiOPRSettings,
        detector: Detector,
    ) -> None:
        self._reconstructor_settings = reconstructor_settings
        self._detector = detector

        self.reconstructor_helper = PtyChiReconstructorOptionsHelper(reconstructor_settings)
        self.object_helper = PtyChiObjectOptionsHelper(object_settings)
        self.probe_helper = PtyChiProbeOptionsHelper(probe_settings)
        self.probe_position_helper = PtyChiProbePositionOptionsHelper(probe_position_settings)
        self.opr_helper = PtyChiOPROptionsHelper(opr_settings)

    def create_data_options(self, parameters: ReconstructInput) -> PtychographyDataOptions:
        metadata = parameters.product.metadata
        return PtychographyDataOptions(
            data=parameters.patterns,
            free_space_propagation_distance_m=metadata.detectorDistanceInMeters,
            wavelength_m=metadata.probeWavelengthInMeters,
            detector_pixel_size_m=self._detector.pixelWidthInMeters.getValue(),
            valid_pixel_mask=parameters.goodPixelMask,
            save_data_on_device=self._reconstructor_settings.saveDataOnDevice.getValue(),
        )
