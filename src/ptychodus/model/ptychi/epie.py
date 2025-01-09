from collections.abc import Sequence
import logging

import numpy

from ptychi.api import (
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
    EPIEOptions,
    EPIEReconstructorOptions,
    ImageGradientMethods,
    ImageIntegrationMethods,
    OPRWeightSmoothingMethods,
    OptimizationPlan,
    Optimizers,
    OrthogonalizationMethods,
    PIEOPRModeWeightsOptions,
    PIEObjectOptions,
    PIEProbeOptions,
    PIEProbePositionOptions,
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
from ptychi.api.task import PtychographyTask

from ptychodus.api.object import Object, ObjectGeometry, ObjectPoint
from ptychodus.api.patterns import BooleanArrayType, DiffractionPatternArrayType
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput, Reconstructor
from ptychodus.api.scan import Scan, ScanPoint

from ..patterns import Detector
from .settings import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)

logger = logging.getLogger(__name__)


class EPIEReconstructor(Reconstructor):
    def __init__(
        self,
        reconstructorSettings: PtyChiReconstructorSettings,
        objectSettings: PtyChiObjectSettings,
        probeSettings: PtyChiProbeSettings,
        probePositionSettings: PtyChiProbePositionSettings,
        oprSettings: PtyChiOPRSettings,
        pieSettings: PtyChiPIESettings,
        detector: Detector,
    ) -> None:
        super().__init__()
        self._reconstructorSettings = reconstructorSettings
        self._objectSettings = objectSettings
        self._probeSettings = probeSettings
        self._probePositionSettings = probePositionSettings
        self._oprSettings = oprSettings
        self._pieSettings = pieSettings
        self._detector = detector

    @property
    def name(self) -> str:
        return 'ePIE'

    def _create_data_options(
        self,
        patterns: DiffractionPatternArrayType,
        goodPixelMask: BooleanArrayType,
        metadata: ProductMetadata,
    ) -> PtychographyDataOptions:
        return PtychographyDataOptions(
            data=patterns,
            free_space_propagation_distance_m=metadata.detectorDistanceInMeters,
            wavelength_m=metadata.probeWavelengthInMeters,
            detector_pixel_size_m=self._detector.pixelWidthInMeters.getValue(),
            valid_pixel_mask=goodPixelMask,
        )

    def _create_reconstructor_options(self) -> EPIEReconstructorOptions:
        batching_mode_str = self._reconstructorSettings.batchingMode.getValue()

        try:
            batching_mode = BatchingModes[batching_mode_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{batching_mode_str}"!')
            batching_mode = BatchingModes.RANDOM

        return EPIEReconstructorOptions(
            num_epochs=self._reconstructorSettings.numEpochs.getValue(),
            batch_size=self._reconstructorSettings.batchSize.getValue(),
            batching_mode=batching_mode,
            compact_mode_update_clustering=self._reconstructorSettings.batchStride.getValue() > 0,
            compact_mode_update_clustering_stride=self._reconstructorSettings.batchStride.getValue(),
            default_device=Devices.GPU
            if self._reconstructorSettings.useDevices.getValue()
            else Devices.CPU,
            default_dtype=(
                Dtypes.FLOAT64
                if self._reconstructorSettings.useDoublePrecision.getValue()
                else Dtypes.FLOAT32
            ),
            # TODO random_seed
            # TODO displayed_loss_function
            use_low_memory_forward_model=self._reconstructorSettings.useLowMemoryForwardModel.getValue(),
        )

    def _create_optimization_plan(self, start: int, stop: int, stride: int) -> OptimizationPlan:
        return OptimizationPlan(start, None if stop < 0 else stop, stride)

    def _create_optimizer(self, text: str) -> Optimizers:
        try:
            optimizer = Optimizers[text.upper()]
        except KeyError:
            logger.warning('Failed to parse optimizer "{text}"!')
            optimizer = Optimizers.SGD

        return optimizer

    def _create_object_options(self, object_: Object) -> PIEObjectOptions:
        optimization_plan = self._create_optimization_plan(
            self._objectSettings.optimizationPlanStart.getValue(),
            self._objectSettings.optimizationPlanStop.getValue(),
            self._objectSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._objectSettings.optimizer.getValue())

        ####

        l1_norm_constraint_optimization_plan = self._create_optimization_plan(
            self._objectSettings.constrainL1NormStart.getValue(),
            self._objectSettings.constrainL1NormStop.getValue(),
            self._objectSettings.constrainL1NormStride.getValue(),
        )
        l1_norm_constraint = ObjectL1NormConstraintOptions(
            enabled=self._objectSettings.constrainL1Norm.getValue(),
            optimization_plan=l1_norm_constraint_optimization_plan,
            weight=self._objectSettings.constrainL1NormWeight.getValue(),
        )

        ####

        smoothness_constraint_optimization_plan = self._create_optimization_plan(
            self._objectSettings.constrainSmoothnessStart.getValue(),
            self._objectSettings.constrainSmoothnessStop.getValue(),
            self._objectSettings.constrainSmoothnessStride.getValue(),
        )
        smoothness_constraint = ObjectSmoothnessConstraintOptions(
            enabled=self._objectSettings.constrainSmoothness.getValue(),
            optimization_plan=smoothness_constraint_optimization_plan,
            alpha=self._objectSettings.constrainSmoothnessAlpha.getValue(),
        )

        ####

        total_variation_optimization_plan = self._create_optimization_plan(
            self._objectSettings.constrainTotalVariationStart.getValue(),
            self._objectSettings.constrainTotalVariationStop.getValue(),
            self._objectSettings.constrainTotalVariationStride.getValue(),
        )
        total_variation = ObjectTotalVariationOptions(
            enabled=self._objectSettings.constrainTotalVariation.getValue(),
            optimization_plan=total_variation_optimization_plan,
            weight=self._objectSettings.constrainTotalVariationWeight.getValue(),
        )

        ####

        remove_grid_artifacts_direction_str = (
            self._objectSettings.removeGridArtifactsDirection.getValue()
        )

        try:
            remove_grid_artifacts_direction = Directions[
                remove_grid_artifacts_direction_str.upper()
            ]
        except KeyError:
            logger.warning('Failed to parse direction "{remove_grid_artifacts_direction_str}"!')
            remove_grid_artifacts_direction = Directions.XY

        remove_grid_artifacts_optimization_plan = self._create_optimization_plan(
            self._objectSettings.removeGridArtifactsStart.getValue(),
            self._objectSettings.removeGridArtifactsStop.getValue(),
            self._objectSettings.removeGridArtifactsStride.getValue(),
        )
        remove_grid_artifacts = RemoveGridArtifactsOptions(
            enabled=self._objectSettings.removeGridArtifacts.getValue(),
            optimization_plan=remove_grid_artifacts_optimization_plan,
            period_x_m=self._objectSettings.removeGridArtifactsPeriodXInMeters.getValue(),
            period_y_m=self._objectSettings.removeGridArtifactsPeriodYInMeters.getValue(),
            window_size=self._objectSettings.removeGridArtifactsWindowSizeInPixels.getValue(),
            direction=remove_grid_artifacts_direction,
        )

        ####

        multislice_regularization_unwrap_image_grad_method_str = (
            self._objectSettings.regularizeMultisliceUnwrapPhaseImageGradientMethod.getValue()
        )

        try:
            multislice_regularization_unwrap_image_grad_method = ImageGradientMethods[
                multislice_regularization_unwrap_image_grad_method_str.upper()
            ]
        except KeyError:
            logger.warning(
                'Failed to parse image gradient method "{multislice_regularization_unwrap_image_grad_method_str}"!'
            )
            multislice_regularization_unwrap_image_grad_method = ImageGradientMethods.FOURIER_SHIFT

        ####

        multislice_regularization_unwrap_image_integration_method_str = (
            self._objectSettings.regularizeMultisliceUnwrapPhaseImageIntegrationMethod.getValue()
        )

        try:
            multislice_regularization_unwrap_image_integration_method = ImageIntegrationMethods[
                multislice_regularization_unwrap_image_integration_method_str.upper()
            ]
        except KeyError:
            logger.warning(
                'Failed to parse image integrationient method "{multislice_regularization_unwrap_image_integration_method_str}"!'
            )
            multislice_regularization_unwrap_image_integration_method = (
                ImageIntegrationMethods.DECONVOLUTION
            )

        multislice_regularization_optimization_plan = self._create_optimization_plan(
            self._objectSettings.regularizeMultisliceStart.getValue(),
            self._objectSettings.regularizeMultisliceStop.getValue(),
            self._objectSettings.regularizeMultisliceStride.getValue(),
        )
        multislice_regularization = ObjectMultisliceRegularizationOptions(
            enabled=self._objectSettings.regularizeMultislice.getValue(),
            optimization_plan=multislice_regularization_optimization_plan,
            weight=self._objectSettings.regularizeMultisliceWeight.getValue(),
            unwrap_phase=self._objectSettings.regularizeMultisliceUnwrapPhase.getValue(),
            unwrap_image_grad_method=multislice_regularization_unwrap_image_grad_method,
            unwrap_image_integration_method=multislice_regularization_unwrap_image_integration_method,
        )

        ####

        patch_interpolation_method_str = self._objectSettings.patchInterpolator.getValue()

        try:
            patch_interpolation_method = PatchInterpolationMethods[
                patch_interpolation_method_str.upper()
            ]
        except KeyError:
            logger.warning(
                'Failed to parse patch interpolation method "{patch_interpolation_method_str}"!'
            )
            patch_interpolation_method = PatchInterpolationMethods.FOURIER

        ####

        pixel_geometry = object_.getPixelGeometry()
        pixel_size_m = 1.0

        if pixel_geometry is None:
            logger.error('Missing object pixel geometry!')
        else:
            pixel_size_m = pixel_geometry.widthInMeters

        ####

        return PIEObjectOptions(
            optimizable=self._objectSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._objectSettings.stepSize.getValue(),
            initial_guess=object_.getArray(),
            slice_spacings_m=numpy.array(object_.layerDistanceInMeters[:-1]),
            pixel_size_m=pixel_size_m,
            l1_norm_constraint=l1_norm_constraint,
            smoothness_constraint=smoothness_constraint,
            total_variation=total_variation,
            remove_grid_artifacts=remove_grid_artifacts,
            multislice_regularization=multislice_regularization,
            patch_interpolation_method=patch_interpolation_method,
            alpha=self._pieSettings.objectAlpha.getValue(),
        )

    def _create_probe_options(self, probe: Probe, metadata: ProductMetadata) -> PIEProbeOptions:
        optimization_plan = self._create_optimization_plan(
            self._probeSettings.optimizationPlanStart.getValue(),
            self._probeSettings.optimizationPlanStop.getValue(),
            self._probeSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._probeSettings.optimizer.getValue())

        ####

        power_constraint_optimization_plan = self._create_optimization_plan(
            self._probeSettings.constrainProbePowerStart.getValue(),
            self._probeSettings.constrainProbePowerStop.getValue(),
            self._probeSettings.constrainProbePowerStride.getValue(),
        )
        power_constraint = ProbePowerConstraintOptions(
            enabled=self._probeSettings.constrainProbePower.getValue(),
            optimization_plan=power_constraint_optimization_plan,
            probe_power=metadata.probePhotonCount,
        )

        ####

        orthogonalize_incoherent_modes_method_str = (
            self._probeSettings.orthogonalizeIncoherentModesMethod.getValue()
        )

        try:
            orthogonalize_incoherent_modes_method = OrthogonalizationMethods[
                orthogonalize_incoherent_modes_method_str.upper()
            ]
        except KeyError:
            logger.warning(
                'Failed to parse batching mode "{orthogonalize_incoherent_modes_method_str}"!'
            )
            orthogonalize_incoherent_modes_method = OrthogonalizationMethods.GS

        orthogonalize_incoherent_modes_optimization_plan = self._create_optimization_plan(
            self._probeSettings.orthogonalizeIncoherentModesStart.getValue(),
            self._probeSettings.orthogonalizeIncoherentModesStop.getValue(),
            self._probeSettings.orthogonalizeIncoherentModesStride.getValue(),
        )
        orthogonalize_incoherent_modes = ProbeOrthogonalizeIncoherentModesOptions(
            enabled=self._probeSettings.orthogonalizeIncoherentModes.getValue(),
            optimization_plan=orthogonalize_incoherent_modes_optimization_plan,
            method=orthogonalize_incoherent_modes_method,
        )

        ####

        orthogonalize_opr_modes_optimization_plan = self._create_optimization_plan(
            self._probeSettings.orthogonalizeOPRModesStart.getValue(),
            self._probeSettings.orthogonalizeOPRModesStop.getValue(),
            self._probeSettings.orthogonalizeOPRModesStride.getValue(),
        )
        orthogonalize_opr_modes = ProbeOrthogonalizeOPRModesOptions(
            enabled=self._probeSettings.orthogonalizeOPRModes.getValue(),
            optimization_plan=orthogonalize_opr_modes_optimization_plan,
        )

        ####

        support_constraint_optimization_plan = self._create_optimization_plan(
            self._probeSettings.constrainSupportStart.getValue(),
            self._probeSettings.constrainSupportStop.getValue(),
            self._probeSettings.constrainSupportStride.getValue(),
        )
        support_constraint = ProbeSupportConstraintOptions(
            enabled=self._probeSettings.constrainSupport.getValue(),
            optimization_plan=support_constraint_optimization_plan,
            threshold=self._probeSettings.constrainSupportThreshold.getValue(),
        )

        ####

        center_constraint_optimization_plan = self._create_optimization_plan(
            self._probeSettings.constrainCenterStart.getValue(),
            self._probeSettings.constrainCenterStop.getValue(),
            self._probeSettings.constrainCenterStride.getValue(),
        )
        center_constraint = ProbeCenterConstraintOptions(
            enabled=self._probeSettings.constrainCenter.getValue(),
            optimization_plan=center_constraint_optimization_plan,
        )

        ####

        return PIEProbeOptions(
            optimizable=self._probeSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._probeSettings.stepSize.getValue(),
            initial_guess=probe.getArray(),
            power_constraint=power_constraint,
            orthogonalize_incoherent_modes=orthogonalize_incoherent_modes,
            orthogonalize_opr_modes=orthogonalize_opr_modes,
            support_constraint=support_constraint,
            center_constraint=center_constraint,
            eigenmode_update_relaxation=self._probeSettings.relaxEigenmodeUpdate.getValue(),
            alpha=self._pieSettings.probeAlpha.getValue(),
        )

    def _create_probe_position_options(
        self, scan: Scan, object_geometry: ObjectGeometry
    ) -> PIEProbePositionOptions:
        probe_position_optimization_plan = self._create_optimization_plan(
            self._probePositionSettings.optimizationPlanStart.getValue(),
            self._probePositionSettings.optimizationPlanStop.getValue(),
            self._probePositionSettings.optimizationPlanStride.getValue(),
        )
        probe_position_optimizer = self._create_optimizer(
            self._probePositionSettings.optimizer.getValue()
        )

        ####

        position_in_px_list: list[float] = list()
        rx_px = object_geometry.widthInPixels / 2
        ry_px = object_geometry.heightInPixels / 2

        for scan_point in scan:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            position_in_px_list.append(object_point.positionYInPixels - ry_px)
            position_in_px_list.append(object_point.positionXInPixels - rx_px)

        position_in_px = numpy.reshape(position_in_px_list, shape=(-1, 2))

        ####

        magnitude_limit_optimization_plan = self._create_optimization_plan(
            self._probePositionSettings.limitMagnitudeUpdateStart.getValue(),
            self._probePositionSettings.limitMagnitudeUpdateStop.getValue(),
            self._probePositionSettings.limitMagnitudeUpdateStride.getValue(),
        )
        magnitude_limit = ProbePositionMagnitudeLimitOptions(
            enabled=self._probePositionSettings.limitMagnitudeUpdate.getValue(),
            optimization_plan=magnitude_limit_optimization_plan,
            limit=self._probePositionSettings.magnitudeUpdateLimit.getValue(),
        )

        ####

        correction_type_str = self._probePositionSettings.positionCorrectionType.getValue()

        try:
            correction_type = PositionCorrectionTypes[correction_type_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{correction_type_str}"!')
            correction_type = PositionCorrectionTypes.GRADIENT

        correction_options = PositionCorrectionOptions(
            correction_type=correction_type,
            cross_correlation_scale=self._probePositionSettings.crossCorrelationScale.getValue(),
            cross_correlation_real_space_width=self._probePositionSettings.crossCorrelationRealSpaceWidth.getValue(),
            cross_correlation_probe_threshold=self._probePositionSettings.crossCorrelationProbeThreshold.getValue(),
        )

        ####

        return PIEProbePositionOptions(
            optimizable=self._probePositionSettings.isOptimizable.getValue(),
            optimization_plan=probe_position_optimization_plan,
            optimizer=probe_position_optimizer,
            step_size=self._probePositionSettings.stepSize.getValue(),
            position_x_px=position_in_px[:, -1],
            position_y_px=position_in_px[:, -2],
            magnitude_limit=magnitude_limit,
            constrain_position_mean=self._probePositionSettings.constrainCentroid.getValue(),
            correction_options=correction_options,
        )

    def _create_opr_mode_weight_options(self) -> PIEOPRModeWeightsOptions:
        opr_optimization_plan = self._create_optimization_plan(
            self._oprSettings.optimizationPlanStart.getValue(),
            self._oprSettings.optimizationPlanStop.getValue(),
            self._oprSettings.optimizationPlanStride.getValue(),
        )
        opr_optimizer = self._create_optimizer(self._oprSettings.optimizer.getValue())

        ####

        smoothing_optimization_plan = self._create_optimization_plan(
            self._oprSettings.smoothModeWeightsStart.getValue(),
            self._oprSettings.smoothModeWeightsStop.getValue(),
            self._oprSettings.smoothModeWeightsStride.getValue(),
        )

        smoothing_method_str = self._oprSettings.smoothingMethod.getValue()

        try:
            smoothing_method: OPRWeightSmoothingMethods | None = OPRWeightSmoothingMethods[
                smoothing_method_str.upper()
            ]
        except KeyError:
            smoothing_method = None
            logger.warning('Failed to parse OPR weight smoothing method "{smoothing_method_str}"!')

        smoothing = OPRModeWeightsSmoothingOptions(
            enabled=self._oprSettings.smoothModeWeights.getValue(),
            optimization_plan=smoothing_optimization_plan,
            method=smoothing_method,
            polynomial_degree=self._oprSettings.polynomialSmoothingDegree.getValue(),
        )

        ####

        return PIEOPRModeWeightsOptions(
            optimizable=self._oprSettings.isOptimizable.getValue(),
            optimization_plan=opr_optimization_plan,
            optimizer=opr_optimizer,
            step_size=self._oprSettings.stepSize.getValue(),
            initial_weights=numpy.array([0.0]),  # FIXME
            optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
            optimize_intensity_variation=self._oprSettings.optimizeIntensities.getValue(),
            smoothing=smoothing,
            update_relaxation=self._oprSettings.relaxUpdate.getValue(),
        )

    def _create_task_options(self, parameters: ReconstructInput) -> EPIEOptions:
        product = parameters.product
        return EPIEOptions(
            data_options=self._create_data_options(
                parameters.patterns, parameters.goodPixelMask, product.metadata
            ),
            reconstructor_options=self._create_reconstructor_options(),
            object_options=self._create_object_options(product.object_),
            probe_options=self._create_probe_options(product.probe, product.metadata),
            probe_position_options=self._create_probe_position_options(
                product.scan, product.object_.getGeometry()
            ),
            opr_mode_weight_options=self._create_opr_mode_weight_options(),
        )

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        task = PtychographyTask(self._create_task_options(parameters))
        # TODO task.iterate(n_epochs)
        task.run()

        # TODO rename to position_x_px and position_y_px
        position_out_px = task.get_data_to_cpu('probe_positions', as_numpy=True)
        probe_out_array = task.get_data_to_cpu('probe', as_numpy=True)
        object_out_array = task.get_data_to_cpu('object', as_numpy=True)
        # TODO opr_mode_weights = task.get_data_to_cpu('opr_mode_weights', as_numpy=True)

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.array(object_out_array),
            layerDistanceInMeters=object_in.layerDistanceInMeters,
            pixelGeometry=object_in.getPixelGeometry(),
            center=object_in.getCenter(),
        )

        probe_in = parameters.product.probe
        probe_out = Probe(
            array=numpy.array(probe_out_array[0]),
            pixelGeometry=probe_in.getPixelGeometry(),
        )

        scan_in = parameters.product.scan
        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.getGeometry()
        rx_px = object_geometry.widthInPixels / 2
        ry_px = object_geometry.heightInPixels / 2

        for uncorrected_point, xy in zip(scan_in, position_out_px):
            object_point = ObjectPoint(
                index=uncorrected_point.index,
                positionXInPixels=xy[-1] + rx_px,
                positionYInPixels=xy[-2] + ry_px,
            )
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = Scan(corrected_scan_points)

        costs: Sequence[float] = list()
        task_reconstructor = task.reconstructor

        if task_reconstructor is not None:
            loss_tracker = task_reconstructor.loss_tracker
            # TODO epoch = loss_tracker.table["epoch"].to_numpy()
            loss = loss_tracker.table['loss'].to_numpy()
            costs = loss  # TODO update api to include epoch and loss

        product = Product(
            metadata=parameters.product.metadata,
            scan=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )

        return ReconstructOutput(product, 0)
