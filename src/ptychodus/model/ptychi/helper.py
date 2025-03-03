from collections.abc import Sequence
import logging

from torch import Tensor
import numpy

from ptychi.api import (
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
    ImageGradientMethods,
    ImageIntegrationMethods,
    LossFunctions,
    OPRWeightSmoothingMethods,
    OptimizationPlan,
    Optimizers,
    OrthogonalizationMethods,
    PatchInterpolationMethods,
    PositionCorrectionTypes,
    PtychographyDataOptions,
)
from ptychi.api.options.base import (
    OPRModeWeightsSmoothingOptions,
    ObjectL1NormConstraintOptions,
    ObjectMultisliceRegularizationOptions,
    ObjectSmoothnessConstraintOptions,
    ObjectTotalVariationOptions,
    PositionCorrectionOptions,
    ProbeCenterConstraintOptions,
    ProbeOrthogonalizeIncoherentModesOptions,
    ProbeOrthogonalizeOPRModesOptions,
    ProbePositionMagnitudeLimitOptions,
    ProbePowerConstraintOptions,
    ProbeSupportConstraintOptions,
    RemoveGridArtifactsOptions,
)

from ptychodus.api.object import Object, ObjectArrayType, ObjectGeometry, ObjectPoint
from ptychodus.api.probe import Probe, WavefieldArrayType
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import Scan, ScanPoint
from ptychodus.api.typing import RealArrayType

from ..patterns import PatternSizer
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

    def get_initial_guess(self, object_: Object) -> ObjectArrayType:
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

    def get_initial_guess(self, probe: Probe) -> WavefieldArrayType:
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

    @property
    def magnitude_limit(self) -> ProbePositionMagnitudeLimitOptions:
        return ProbePositionMagnitudeLimitOptions(
            enabled=self._settings.limitMagnitudeUpdate.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.limitMagnitudeUpdateStart.getValue(),
                self._settings.limitMagnitudeUpdateStop.getValue(),
                self._settings.limitMagnitudeUpdateStride.getValue(),
            ),
            limit=self._settings.magnitudeUpdateLimit.getValue(),
        )

    @property
    def constrain_position_mean(self) -> bool:
        return self._settings.constrainCentroid.getValue()

    @property
    def correction_options(self) -> PositionCorrectionOptions:
        correction_type_str = self._settings.positionCorrectionType.getValue()

        try:
            correction_type = PositionCorrectionTypes[correction_type_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{correction_type_str}"!')
            correction_type = PositionCorrectionTypes.GRADIENT

        return PositionCorrectionOptions(
            correction_type=correction_type,
            cross_correlation_scale=self._settings.crossCorrelationScale.getValue(),
            cross_correlation_real_space_width=self._settings.crossCorrelationRealSpaceWidth.getValue(),
            cross_correlation_probe_threshold=self._settings.crossCorrelationProbeThreshold.getValue(),
        )

    def get_positions_px(
        self, scan: Scan, object_geometry: ObjectGeometry
    ) -> tuple[RealArrayType, RealArrayType]:
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()
        rx_px = object_geometry.widthInPixels / 2
        ry_px = object_geometry.heightInPixels / 2

        for scan_point in scan:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            position_x_px.append(object_point.positionXInPixels - rx_px)
            position_y_px.append(object_point.positionYInPixels - ry_px)

        return numpy.array(position_x_px), numpy.array(position_y_px)


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

    @property
    def smoothing(self) -> OPRModeWeightsSmoothingOptions:
        method_str = self._settings.smoothingMethod.getValue()

        try:
            method: OPRWeightSmoothingMethods | None = OPRWeightSmoothingMethods[method_str.upper()]
        except KeyError:
            method = None
            logger.warning('Failed to parse OPR weight smoothing method "{method_str}"!')

        return OPRModeWeightsSmoothingOptions(
            enabled=self._settings.smoothModeWeights.getValue(),
            optimization_plan=create_optimization_plan(
                self._settings.smoothModeWeightsStart.getValue(),
                self._settings.smoothModeWeightsStop.getValue(),
                self._settings.smoothModeWeightsStride.getValue(),
            ),
            method=method,
            polynomial_degree=self._settings.polynomialSmoothingDegree.getValue(),
        )

    @property
    def optimize_eigenmode_weights(self) -> bool:
        return self._settings.optimizeEigenmodeWeights.getValue()

    @property
    def optimize_intensity_variation(self) -> bool:
        return self._settings.optimizeIntensities.getValue()

    @property
    def update_relaxation(self) -> float:
        return self._settings.relaxUpdate.getValue()

    def get_initial_weights(self) -> RealArrayType:
        return numpy.array([0.0])  # FIXME


class PtyChiOptionsHelper:
    def __init__(
        self,
        reconstructor_settings: PtyChiReconstructorSettings,
        object_settings: PtyChiObjectSettings,
        probe_settings: PtyChiProbeSettings,
        probe_position_settings: PtyChiProbePositionSettings,
        opr_settings: PtyChiOPRSettings,
        pattern_sizer: PatternSizer,
    ) -> None:
        self._reconstructor_settings = reconstructor_settings
        self._pattern_sizer = pattern_sizer

        self.reconstructor_helper = PtyChiReconstructorOptionsHelper(reconstructor_settings)
        self.object_helper = PtyChiObjectOptionsHelper(object_settings)
        self.probe_helper = PtyChiProbeOptionsHelper(probe_settings)
        self.probe_position_helper = PtyChiProbePositionOptionsHelper(probe_position_settings)
        self.opr_helper = PtyChiOPROptionsHelper(opr_settings)

    def create_data_options(self, parameters: ReconstructInput) -> PtychographyDataOptions:
        metadata = parameters.product.metadata
        pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
        return PtychographyDataOptions(
            data=parameters.patterns,
            free_space_propagation_distance_m=metadata.detectorDistanceInMeters,
            wavelength_m=metadata.probeWavelengthInMeters,
            detector_pixel_size_m=pixel_geometry.widthInMeters,
            valid_pixel_mask=numpy.logical_not(parameters.bad_pixels),
            save_data_on_device=self._reconstructor_settings.saveDataOnDevice.getValue(),
        )

    def create_product(
        self,
        product: Product,
        position_x_px: Tensor | numpy.ndarray,
        position_y_px: Tensor | numpy.ndarray,
        probe_array: Tensor | numpy.ndarray,
        object_array: Tensor | numpy.ndarray,
        opr_mode_weights: Tensor | numpy.ndarray,
        costs: Sequence[float],
    ) -> Product:
        object_in = product.object_
        object_out = Object(
            array=numpy.array(object_array),
            layerDistanceInMeters=object_in.layerDistanceInMeters,
            pixelGeometry=object_in.getPixelGeometry(),
            center=object_in.getCenter(),
        )

        # TODO OPR
        probe_out = Probe(
            array=numpy.array(probe_array[0]),
            pixelGeometry=product.probe.getPixelGeometry(),
        )

        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.getGeometry()
        rx_px = object_geometry.widthInPixels / 2
        ry_px = object_geometry.heightInPixels / 2

        for uncorrected_point, pos_x_px, pos_y_px in zip(
            product.scan, position_x_px, position_y_px
        ):
            object_point = ObjectPoint(
                index=uncorrected_point.index,
                positionXInPixels=pos_x_px + rx_px,
                positionYInPixels=pos_y_px + ry_px,
            )
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = Scan(corrected_scan_points)

        return Product(
            metadata=product.metadata,
            scan=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )
