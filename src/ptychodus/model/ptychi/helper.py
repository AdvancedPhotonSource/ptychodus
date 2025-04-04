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
    ForwardModelOptions,
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
    RemoveObjectProbeAmbiguityOptions,
)

from ptychodus.api.object import Object, ObjectArrayType, ObjectGeometry, ObjectPoint
from ptychodus.api.probe import Probe, WavefieldArrayType
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import PositionSequence, ScanPoint
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
        return self._settings.num_epochs.get_value()

    @property
    def batch_size(self) -> int:
        return self._settings.batch_size.get_value()

    @property
    def batching_mode(self) -> BatchingModes:
        batching_mode_str = self._settings.batching_mode.get_value()

        try:
            return BatchingModes[batching_mode_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{batching_mode_str}"!')
            return BatchingModes.RANDOM

    @property
    def compact_mode_update_clustering(self) -> bool:
        return self._settings.compact_mode_update_clustering.get_value() > 0

    @property
    def compact_mode_update_clustering_stride(self) -> int:
        return self._settings.compact_mode_update_clustering.get_value()

    @property
    def default_device(self) -> Devices:
        return Devices.GPU if self._settings.use_devices.get_value() else Devices.CPU

    @property
    def default_dtype(self) -> Dtypes:
        return Dtypes.FLOAT64 if self._settings.use_double_precision.get_value() else Dtypes.FLOAT32

    @property
    def random_seed(self) -> int | None:
        return None  # TODO

    @property
    def displayed_loss_function(self) -> LossFunctions | None:
        return LossFunctions.MSE_SQRT  # TODO

    @property
    def forward_model_options(self) -> ForwardModelOptions:
        return ForwardModelOptions(
            low_memory_mode=self._settings.use_low_memory_mode.get_value(),
            pad_for_shift=self._settings.pad_for_shift.get_value(),
        )


class PtyChiObjectOptionsHelper:
    def __init__(self, settings: PtyChiObjectSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.is_optimizable.get_value()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimization_plan_start.get_value(),
            self._settings.optimization_plan_stop.get_value(),
            self._settings.optimization_plan_stride.get_value(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.get_value())

    @property
    def step_size(self) -> float:
        return self._settings.step_size.get_value()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def l1_norm_constraint(self) -> ObjectL1NormConstraintOptions:
        return ObjectL1NormConstraintOptions(
            enabled=self._settings.constrain_l1_norm.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_l1_norm_start.get_value(),
                self._settings.constrain_l1_norm_stop.get_value(),
                self._settings.constrain_l1_norm_stride.get_value(),
            ),
            weight=self._settings.constrain_l1_norm_weight.get_value(),
        )

    @property
    def smoothness_constraint(self) -> ObjectSmoothnessConstraintOptions:
        return ObjectSmoothnessConstraintOptions(
            enabled=self._settings.constrain_smoothness.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_smoothness_start.get_value(),
                self._settings.constrain_smoothness_stop.get_value(),
                self._settings.constrain_smoothness_stride.get_value(),
            ),
            alpha=self._settings.constrain_smoothness_alpha.get_value(),
        )

    @property
    def total_variation(self) -> ObjectTotalVariationOptions:
        return ObjectTotalVariationOptions(
            enabled=self._settings.constrain_total_variation.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_total_variation_start.get_value(),
                self._settings.constrain_total_variation_stop.get_value(),
                self._settings.constrain_total_variation_stride.get_value(),
            ),
            weight=self._settings.constrain_total_variation_weight.get_value(),
        )

    @property
    def remove_grid_artifacts(self) -> RemoveGridArtifactsOptions:
        direction_str = self._settings.remove_grid_artifacts_direction.get_value()

        try:
            direction = Directions[direction_str.upper()]
        except KeyError:
            logger.warning('Failed to parse direction "{direction_str}"!')
            direction = Directions.XY

        return RemoveGridArtifactsOptions(
            enabled=self._settings.remove_grid_artifacts.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.remove_grid_artifacts_start.get_value(),
                self._settings.remove_grid_artifacts_stop.get_value(),
                self._settings.remove_grid_artifacts_stride.get_value(),
            ),
            period_x_m=self._settings.remove_grid_artifacts_period_x_m.get_value(),
            period_y_m=self._settings.remove_grid_artifacts_period_y_m.get_value(),
            window_size=self._settings.remove_grid_artifacts_window_size_px.get_value(),
            direction=direction,
        )

    @property
    def multislice_regularization(self) -> ObjectMultisliceRegularizationOptions:
        unwrap_image_grad_method_str = (
            self._settings.regularize_multislice_unwrap_phase_image_gradient_method.get_value()
        )

        try:
            unwrap_image_grad_method = ImageGradientMethods[unwrap_image_grad_method_str.upper()]
        except KeyError:
            logger.warning(
                'Failed to parse image gradient method "{unwrap_image_grad_method_str}"!'
            )
            unwrap_image_grad_method = ImageGradientMethods.FOURIER_SHIFT

        unwrap_image_integration_method_str = (
            self._settings.regularize_multislice_unwrap_phase_image_integration_method.get_value()
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
            enabled=self._settings.regularize_multislice.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.regularize_multislice_start.get_value(),
                self._settings.regularize_multislice_stop.get_value(),
                self._settings.regularize_multislice_stride.get_value(),
            ),
            weight=self._settings.regularize_multislice_weight.get_value(),
            unwrap_phase=self._settings.regularize_multislice_unwrap_phase.get_value(),
            unwrap_image_grad_method=unwrap_image_grad_method,
            unwrap_image_integration_method=unwrap_image_integration_method,
        )

    @property
    def patch_interpolation_method(self) -> PatchInterpolationMethods:
        method_str = self._settings.patch_interpolator.get_value()

        try:
            return PatchInterpolationMethods[method_str.upper()]
        except KeyError:
            logger.warning('Failed to parse patch interpolation method "{method_str}"!')
            return PatchInterpolationMethods.FOURIER

    @property
    def remove_object_probe_ambiguity(self) -> RemoveObjectProbeAmbiguityOptions:
        return RemoveObjectProbeAmbiguityOptions(
            enabled=self._settings.remove_object_probe_ambiguity.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.remove_object_probe_ambiguity_start.get_value(),
                self._settings.remove_object_probe_ambiguity_stop.get_value(),
                self._settings.remove_object_probe_ambiguity_stride.get_value(),
            ),
        )

    @property
    def build_preconditioner_with_all_modes(self) -> bool:
        return self._settings.build_preconditioner_with_all_modes.get_value()

    def get_initial_guess(self, object_: Object) -> ObjectArrayType:
        return object_.get_array()

    def get_slice_spacings_m(self, object_: Object) -> RealArrayType:
        return numpy.array(object_.layer_distance_m[:-1])  # FIXME iff multislice

    def get_pixel_size_m(self, object_: Object) -> float:
        pixel_geometry = object_.get_pixel_geometry()

        if pixel_geometry is None:
            logger.error('Missing object pixel geometry!')
            return 1.0

        return pixel_geometry.width_m


class PtyChiProbeOptionsHelper:
    def __init__(self, settings: PtyChiProbeSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.is_optimizable.get_value()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimization_plan_start.get_value(),
            self._settings.optimization_plan_stop.get_value(),
            self._settings.optimization_plan_stride.get_value(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.get_value())

    @property
    def step_size(self) -> float:
        return self._settings.step_size.get_value()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def orthogonalize_incoherent_modes(self) -> ProbeOrthogonalizeIncoherentModesOptions:
        method_str = self._settings.orthogonalize_incoherent_modes_method.get_value()

        try:
            method = OrthogonalizationMethods[method_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{method_str}"!')
            method = OrthogonalizationMethods.GS

        return ProbeOrthogonalizeIncoherentModesOptions(
            enabled=self._settings.orthogonalize_incoherent_modes.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.orthogonalize_incoherent_modes_start.get_value(),
                self._settings.orthogonalize_incoherent_modes_stop.get_value(),
                self._settings.orthogonalize_incoherent_modes_stride.get_value(),
            ),
            method=method,
        )

    @property
    def orthogonalize_opr_modes(self) -> ProbeOrthogonalizeOPRModesOptions:
        return ProbeOrthogonalizeOPRModesOptions(
            enabled=self._settings.orthogonalize_opr_modes.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.orthogonalize_opr_modes_start.get_value(),
                self._settings.orthogonalize_opr_modes_stop.get_value(),
                self._settings.orthogonalize_opr_modes_stride.get_value(),
            ),
        )

    @property
    def support_constraint(self) -> ProbeSupportConstraintOptions:
        return ProbeSupportConstraintOptions(
            enabled=self._settings.constrain_support.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_support_start.get_value(),
                self._settings.constrain_support_stop.get_value(),
                self._settings.constrain_support_stride.get_value(),
            ),
            threshold=self._settings.constrain_support_threshold.get_value(),
        )

    @property
    def center_constraint(self) -> ProbeCenterConstraintOptions:
        return ProbeCenterConstraintOptions(
            enabled=self._settings.constrain_center.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_center_start.get_value(),
                self._settings.constrain_center_stop.get_value(),
                self._settings.constrain_center_stride.get_value(),
            ),
        )

    @property
    def eigenmode_update_relaxation(self) -> float:
        return self._settings.relax_eigenmode_update.get_value()

    def get_initial_guess(self, probe: Probe) -> WavefieldArrayType:
        return probe.get_array()

    def get_power_constraint(self, metadata: ProductMetadata) -> ProbePowerConstraintOptions:
        return ProbePowerConstraintOptions(
            enabled=self._settings.constrain_probe_power.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_probe_power_start.get_value(),
                self._settings.constrain_probe_power_stop.get_value(),
                self._settings.constrain_probe_power_stride.get_value(),
            ),
            probe_power=metadata.probe_photon_count,
        )


class PtyChiProbePositionOptionsHelper:
    def __init__(self, settings: PtyChiProbePositionSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.is_optimizable.get_value()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimization_plan_start.get_value(),
            self._settings.optimization_plan_stop.get_value(),
            self._settings.optimization_plan_stride.get_value(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.get_value())

    @property
    def step_size(self) -> float:
        return self._settings.step_size.get_value()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def magnitude_limit(self) -> ProbePositionMagnitudeLimitOptions:
        return ProbePositionMagnitudeLimitOptions(
            enabled=self._settings.limit_magnitude_update.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.limit_magnitude_update_start.get_value(),
                self._settings.limit_magnitude_update_stop.get_value(),
                self._settings.limit_magnitude_update_stride.get_value(),
            ),
            limit=self._settings.magnitude_update_limit.get_value(),
        )

    @property
    def constrain_position_mean(self) -> bool:
        return self._settings.constrain_centroid.get_value()

    @property
    def correction_options(self) -> PositionCorrectionOptions:
        correction_type_str = self._settings.position_correction_type.get_value()

        try:
            correction_type = PositionCorrectionTypes[correction_type_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{correction_type_str}"!')
            correction_type = PositionCorrectionTypes.GRADIENT

        return PositionCorrectionOptions(
            correction_type=correction_type,
            cross_correlation_scale=self._settings.cross_correlation_scale.get_value(),
            cross_correlation_real_space_width=self._settings.cross_correlation_real_space_width.get_value(),
            cross_correlation_probe_threshold=self._settings.cross_correlation_probe_threshold.get_value(),
        )

    def get_positions_px(
        self, scan: PositionSequence, object_geometry: ObjectGeometry
    ) -> tuple[RealArrayType, RealArrayType]:
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()
        rx_px = object_geometry.width_px / 2
        ry_px = object_geometry.height_px / 2

        for scan_point in scan:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            position_x_px.append(object_point.position_x_px - rx_px)
            position_y_px.append(object_point.position_y_px - ry_px)

        return numpy.array(position_x_px), numpy.array(position_y_px)


class PtyChiOPROptionsHelper:
    def __init__(self, settings: PtyChiOPRSettings) -> None:
        self._settings = settings

    @property
    def optimizable(self) -> bool:
        return self._settings.is_optimizable.get_value()

    @property
    def optimization_plan(self) -> OptimizationPlan:
        return create_optimization_plan(
            self._settings.optimization_plan_start.get_value(),
            self._settings.optimization_plan_stop.get_value(),
            self._settings.optimization_plan_stride.get_value(),
        )

    @property
    def optimizer(self) -> Optimizers:
        return parse_optimizer(self._settings.optimizer.get_value())

    @property
    def step_size(self) -> float:
        return self._settings.step_size.get_value()

    @property
    def optimizer_params(self) -> dict:  # TODO
        return dict()

    @property
    def smoothing(self) -> OPRModeWeightsSmoothingOptions:
        method_str = self._settings.smoothing_method.get_value()

        try:
            method: OPRWeightSmoothingMethods | None = OPRWeightSmoothingMethods[method_str.upper()]
        except KeyError:
            method = None
            logger.warning('Failed to parse OPR weight smoothing method "{method_str}"!')

        return OPRModeWeightsSmoothingOptions(
            enabled=self._settings.smooth_mode_weights.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.smooth_mode_weights_start.get_value(),
                self._settings.smooth_mode_weights_stop.get_value(),
                self._settings.smooth_mode_weights_stride.get_value(),
            ),
            method=method,
            polynomial_degree=self._settings.polynomial_smoothing_degree.get_value(),
        )

    @property
    def optimize_eigenmode_weights(self) -> bool:
        return self._settings.optimize_eigenmode_weights.get_value()

    @property
    def optimize_intensity_variation(self) -> bool:
        return self._settings.optimize_intensities.get_value()

    @property
    def update_relaxation(self) -> float:
        return self._settings.relax_update.get_value()

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
        free_space_propagation_distance_m = (
            numpy.inf
            if self._reconstructor_settings.use_far_field_propagation
            else metadata.detector_distance_m
        )
        return PtychographyDataOptions(
            data=parameters.patterns,
            free_space_propagation_distance_m=free_space_propagation_distance_m,
            wavelength_m=metadata.probe_wavelength_m,
            fft_shift=self._reconstructor_settings.fft_shift_diffraction_patterns.get_value(),
            detector_pixel_size_m=pixel_geometry.width_m,
            valid_pixel_mask=numpy.logical_not(parameters.bad_pixels),
            save_data_on_device=self._reconstructor_settings.save_data_on_device.get_value(),
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
            layer_distance_m=object_in.layer_distance_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )

        # TODO OPR
        probe_out = Probe(
            array=numpy.array(probe_array[0]),
            pixel_geometry=product.probe.get_pixel_geometry(),
        )

        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.get_geometry()
        rx_px = object_geometry.width_px / 2
        ry_px = object_geometry.height_px / 2

        for uncorrected_point, pos_x_px, pos_y_px in zip(
            product.positions, position_x_px, position_y_px
        ):
            object_point = ObjectPoint(
                index=uncorrected_point.index,
                position_x_px=pos_x_px + rx_px,
                position_y_px=pos_y_px + ry_px,
            )
            scan_point = object_geometry.map_object_point_to_scan_point(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = PositionSequence(corrected_scan_points)

        return Product(
            metadata=product.metadata,
            positions=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )
