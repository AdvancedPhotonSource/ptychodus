from collections.abc import Sequence
import logging

import torch
import numpy

from ptychi.api import (
    AffineDegreesOfFreedom,
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
    ImageGradientMethods,
    ImageIntegrationMethods,
    LossFunctions,
    OPRWeightSmoothingMethods,
    ObjectPosOriginCoordsMethods,
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
    ObjectL2NormConstraintOptions,
    ObjectMultisliceRegularizationOptions,
    ObjectSmoothnessConstraintOptions,
    ObjectTotalVariationOptions,
    OptimizationPlan,
    PositionAffineTransformConstraintOptions,
    PositionCorrectionOptions,
    ProbeCenterConstraintOptions,
    ProbeOrthogonalizeIncoherentModesOptions,
    ProbeOrthogonalizeOPRModesOptions,
    ProbePowerConstraintOptions,
    ProbeSupportConstraintOptions,
    RemoveGridArtifactsOptions,
    RemoveObjectProbeAmbiguityOptions,
    SliceSpacingOptions,
)

from ptychodus.api.object import Object, ObjectGeometry, ObjectPoint
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, Product, ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import PositionSequence, ScanPoint
from ptychodus.api.typing import ComplexArrayType, RealArrayType

from ..diffraction import PatternSizer
from .affine import PtyChiAffineDegreesOfFreedom, PtyChiAffineDegreesOfFreedomBitField
from .settings import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiSettings,
)


__all__ = ['PtyChiOptionsHelper']

logger = logging.getLogger(__name__)


def create_optimization_plan(start: int, stop: int, stride: int) -> OptimizationPlan:
    return OptimizationPlan(start, None if stop < 0 else stop, stride)


def parse_optimizer(text: str) -> Optimizers:
    try:
        optimizer = Optimizers[text.upper()]
    except KeyError:
        logger.warning(f'Failed to parse optimizer "{text}"!')
        optimizer = Optimizers.SGD

    return optimizer


class PtyChiReconstructorOptionsHelper:
    def __init__(self, settings: PtyChiSettings) -> None:
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
            logger.warning(f'Failed to parse batching mode "{batching_mode_str}"!')
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
    def use_double_precision_for_fft(self) -> bool:
        return self._settings.use_double_precision_for_fft.get_value()

    @property
    def allow_nondeterministic_algorithms(self) -> bool:
        return self._settings.allow_nondeterministic_algorithms.get_value()

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
    def slice_spacing_options(self) -> SliceSpacingOptions:
        optimizer = parse_optimizer(self._settings.optimize_slice_spacing_optimizer.get_value())
        return SliceSpacingOptions(
            optimizable=self._settings.optimize_slice_spacing.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.optimize_slice_spacing_start.get_value(),
                self._settings.optimize_slice_spacing_stop.get_value(),
                self._settings.optimize_slice_spacing_stride.get_value(),
            ),
            optimizer=optimizer,
            step_size=self._settings.optimize_slice_spacing_step_size.get_value(),
        )

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
    def l2_norm_constraint(self) -> ObjectL2NormConstraintOptions:
        return ObjectL2NormConstraintOptions(
            enabled=self._settings.constrain_l2_norm.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_l2_norm_start.get_value(),
                self._settings.constrain_l2_norm_stop.get_value(),
                self._settings.constrain_l2_norm_stride.get_value(),
            ),
            weight=self._settings.constrain_l2_norm_weight.get_value(),
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
            logger.warning(f'Failed to parse direction "{direction_str}"!')
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
                f'Failed to parse image gradient method "{unwrap_image_grad_method_str}"!'
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
                f'Failed to parse image integrationient method "{unwrap_image_integration_method_str}"!'
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
            logger.warning(f'Failed to parse patch interpolation method "{method_str}"!')
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

    @property
    def determine_position_origin_coords_by(self) -> ObjectPosOriginCoordsMethods:
        return ObjectPosOriginCoordsMethods.SPECIFIED

    def get_initial_guess(self, object_: Object) -> ComplexArrayType:
        return object_.get_array()

    def get_slice_spacings_m(self, object_: Object) -> RealArrayType | None:
        slice_spacings_m = object_.layer_spacing_m
        return numpy.array(slice_spacings_m) if slice_spacings_m else None

    def get_pixel_size_m(self, object_: Object) -> float:
        pixel_geometry = object_.get_pixel_geometry()
        return pixel_geometry.width_m

    def get_pixel_aspect_ratio(self, object_: Object) -> float:
        pixel_geometry = object_.get_pixel_geometry()
        return pixel_geometry.aspect_ratio

    def get_position_origin_coords(self, object_: Object) -> RealArrayType:
        # TODO return numpy.zeros(2)
        return torch.zeros(2)  # type: ignore


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
            logger.warning(f'Failed to parse batching mode "{method_str}"!')
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
            use_intensity_for_com=self._settings.use_intensity_for_mass_centroid.get_value(),
        )

    @property
    def eigenmode_update_relaxation(self) -> float:
        return self._settings.relax_eigenmode_update.get_value()

    def get_initial_guess(self, probe: ProbeSequence) -> ComplexArrayType:
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
        self._affine_dof = PtyChiAffineDegreesOfFreedomBitField(
            settings.constrain_affine_transform_degrees_of_freedom
        )

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
    def constrain_position_mean(self) -> bool:
        return self._settings.constrain_centroid.get_value()

    @property
    def correction_options(self) -> PositionCorrectionOptions:
        correction_type_str = self._settings.correction_type.get_value()

        try:
            correction_type = PositionCorrectionTypes[correction_type_str.upper()]
        except KeyError:
            logger.warning(f'Failed to parse correction type "{correction_type_str}"!')
            correction_type = PositionCorrectionTypes.GRADIENT

        differentiation_method_str = self._settings.differentiation_method.get_value()

        try:
            differentiation_method = ImageGradientMethods[differentiation_method_str.upper()]
        except KeyError:
            logger.warning(
                f'Failed to parse differentiation method "{differentiation_method_str}"!'
            )
            differentiation_method = ImageGradientMethods.FOURIER_DIFFERENTIATION

        slice_for_correction = (
            self._settings.slice_for_correction.get_value()
            if self._settings.choose_slice_for_correction.get_value()
            else None
        )
        update_magnitude_limit = (
            self._settings.update_magnitude_limit.get_value()
            if self._settings.limit_update_magnitude.get_value()
            else None
        )

        return PositionCorrectionOptions(
            correction_type=correction_type,
            differentiation_method=differentiation_method,
            cross_correlation_scale=self._settings.cross_correlation_scale.get_value(),
            cross_correlation_real_space_width=self._settings.cross_correlation_real_space_width.get_value(),
            cross_correlation_probe_threshold=self._settings.cross_correlation_probe_threshold.get_value(),
            slice_for_correction=slice_for_correction,  # type: ignore
            clip_update_magnitude_by_mad=self._settings.clip_update_magnitude_by_mad.get_value(),
            update_magnitude_limit=update_magnitude_limit,
        )

    @property
    def affine_transform_constraint(self) -> PositionAffineTransformConstraintOptions:
        degrees_of_freedom: list[AffineDegreesOfFreedom] = list()

        if self._affine_dof.is_bit_set(PtyChiAffineDegreesOfFreedom.TRANSLATION):
            degrees_of_freedom.append(AffineDegreesOfFreedom.TRANSLATION)

        if self._affine_dof.is_bit_set(PtyChiAffineDegreesOfFreedom.ROTATION):
            degrees_of_freedom.append(AffineDegreesOfFreedom.ROTATION)

        if self._affine_dof.is_bit_set(PtyChiAffineDegreesOfFreedom.SCALING):
            degrees_of_freedom.append(AffineDegreesOfFreedom.SCALE)

        if self._affine_dof.is_bit_set(PtyChiAffineDegreesOfFreedom.SHEARING):
            degrees_of_freedom.append(AffineDegreesOfFreedom.SHEAR)

        if self._affine_dof.is_bit_set(PtyChiAffineDegreesOfFreedom.ASYMMETRY):
            degrees_of_freedom.append(AffineDegreesOfFreedom.ASYMMETRY)

        return PositionAffineTransformConstraintOptions(
            enabled=self._settings.constrain_affine_transform.get_value(),
            optimization_plan=create_optimization_plan(
                self._settings.constrain_affine_transform_start.get_value(),
                self._settings.constrain_affine_transform_stop.get_value(),
                self._settings.constrain_affine_transform_stride.get_value(),
            ),
            degrees_of_freedom=degrees_of_freedom,
            position_weight_update_interval=self._settings.constrain_affine_transform_position_weight_update_interval.get_value(),
            apply_constraint=self._settings.constrain_affine_transform_apply_constraint.get_value(),
            max_expected_error=self._settings.constrain_affine_transform_max_expected_error_px.get_value(),
        )

    def get_positions_px(
        self, scan: PositionSequence, object_geometry: ObjectGeometry
    ) -> tuple[RealArrayType, RealArrayType]:
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()

        for scan_point in scan:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            position_x_px.append(object_point.position_x_px)
            position_y_px.append(object_point.position_y_px)

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
    def optimize_eigenmode_weights(self) -> bool:
        return self._settings.optimize_eigenmode_weights.get_value()

    @property
    def optimize_intensity_variation(self) -> bool:
        return self._settings.optimize_intensities.get_value()

    @property
    def smoothing(self) -> OPRModeWeightsSmoothingOptions:
        method_str = self._settings.smoothing_method.get_value()

        try:
            method: OPRWeightSmoothingMethods = OPRWeightSmoothingMethods[method_str.upper()]
        except KeyError:
            logger.debug('OPR weight smoothing method is None.')
            method = OPRWeightSmoothingMethods.MEDIAN

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
    def update_relaxation(self) -> float:
        return self._settings.relax_update.get_value()

    def get_initial_weights(self, probe: ProbeSequence) -> RealArrayType:
        try:
            return probe.get_opr_weights()
        except ValueError:
            pass

        initial_weights = numpy.zeros((probe.num_coherent_modes))
        initial_weights[0] = 1.0
        return initial_weights


class PtyChiOptionsHelper:
    def __init__(
        self,
        reconstructor_settings: PtyChiSettings,
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
            data=parameters.diffraction_patterns,
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
        position_x_px: torch.Tensor | numpy.ndarray,
        position_y_px: torch.Tensor | numpy.ndarray,
        probe_array: torch.Tensor | numpy.ndarray,
        object_array: torch.Tensor | numpy.ndarray,
        opr_weights: torch.Tensor | numpy.ndarray,
        losses: Sequence[LossValue],
    ) -> Product:
        object_in = product.object_
        object_out = Object(
            array=numpy.array(object_array),
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )

        probe_out = ProbeSequence(
            array=numpy.array(probe_array[0]),
            opr_weights=numpy.array(opr_weights),
            pixel_geometry=product.probes.get_pixel_geometry(),
        )

        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.get_geometry()

        for uncorrected_point, pos_x_px, pos_y_px in zip(
            product.positions, position_x_px, position_y_px
        ):
            object_point = ObjectPoint(
                index=uncorrected_point.index,
                position_x_px=float(pos_x_px),
                position_y_px=float(pos_y_px),
            )
            scan_point = object_geometry.map_object_point_to_scan_point(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = PositionSequence(corrected_scan_points)

        return Product(
            metadata=product.metadata,
            positions=scan_out,
            probes=probe_out,
            object_=object_out,
            losses=losses,
        )
