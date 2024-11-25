from collections.abc import Sequence
import logging

import numpy

from ptychi.api import (
    BatchingModes,
    DMOPRModeWeightsOptions,
    DMObjectOptions,
    DMOptions,
    DMProbeOptions,
    DMProbePositionOptions,
    DMReconstructorOptions,
    Devices,
    Directions,
    Dtypes,
    ImageGradientMethods,
    OptimizationPlan,
    Optimizers,
    OrthogonalizationMethods,
    PatchInterpolationMethods,
    PtychographyDataOptions,
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
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)

logger = logging.getLogger(__name__)


class DMReconstructor(Reconstructor):
    def __init__(
        self,
        reconstructorSettings: PtyChiReconstructorSettings,
        objectSettings: PtyChiObjectSettings,
        probeSettings: PtyChiProbeSettings,
        probePositionSettings: PtyChiProbePositionSettings,
        oprSettings: PtyChiOPRSettings,
        detector: Detector,
    ) -> None:
        super().__init__()
        self._reconstructorSettings = reconstructorSettings
        self._objectSettings = objectSettings
        self._probeSettings = probeSettings
        self._probePositionSettings = probePositionSettings
        self._oprSettings = oprSettings
        self._detector = detector

    @property
    def name(self) -> str:
        return 'DM'

    def _create_data_options(
        self,
        patterns: DiffractionPatternArrayType,
        goodPixelMask: BooleanArrayType,
        metadata: ProductMetadata,
    ) -> PtychographyDataOptions:
        return PtychographyDataOptions(
            data=patterns,
            propagation_distance_m=metadata.detectorDistanceInMeters,
            wavelength_m=metadata.probeWavelengthInMeters,
            detector_pixel_size_m=self._detector.pixelWidthInMeters.getValue(),
            valid_pixel_mask=goodPixelMask,
        )

    def _create_reconstructor_options(self) -> DMReconstructorOptions:
        batching_mode_str = self._reconstructorSettings.batchingMode.getValue()

        try:
            batching_mode = BatchingModes[batching_mode_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{batching_mode_str}"!')
            batching_mode = BatchingModes.RANDOM

        return DMReconstructorOptions(
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

    def _create_object_options(self, object_: Object) -> DMObjectOptions:
        optimization_plan = self._create_optimization_plan(
            self._objectSettings.optimizationPlanStart.getValue(),
            self._objectSettings.optimizationPlanStop.getValue(),
            self._objectSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._objectSettings.optimizer.getValue())

        ####

        l1_norm_constraint_weight = (
            self._objectSettings.constrainL1NormWeight.getValue()
            if self._objectSettings.constrainL1Norm.getValue()
            else 0.0
        )
        smoothness_constraint_alpha = (
            self._objectSettings.constrainSmoothnessAlpha.getValue()
            if self._objectSettings.constrainSmoothness.getValue()
            else 0.0
        )
        total_variation_weight = (
            self._objectSettings.constrainTotalVariationWeight.getValue()
            if self._objectSettings.constrainTotalVariation.getValue()
            else 0.0
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

        ####

        multislice_regularization_weight = (
            self._objectSettings.regularizeMultisliceWeight.getValue()
            if self._objectSettings.regularizeMultislice.getValue()
            else 0.0
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

        return DMObjectOptions(
            optimizable=self._objectSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._objectSettings.stepSize.getValue(),
            initial_guess=object_.getArray(),
            slice_spacings_m=numpy.array(object_.layerDistanceInMeters[:-1]),
            l1_norm_constraint_weight=l1_norm_constraint_weight,
            l1_norm_constraint_stride=self._objectSettings.constrainL1NormStride.getValue(),
            smoothness_constraint_alpha=smoothness_constraint_alpha,
            smoothness_constraint_stride=self._objectSettings.constrainSmoothnessStride.getValue(),
            total_variation_weight=total_variation_weight,
            total_variation_stride=self._objectSettings.constrainTotalVariationStride.getValue(),
            remove_grid_artifacts=self._objectSettings.removeGridArtifacts.getValue(),
            remove_grid_artifacts_period_x_m=self._objectSettings.removeGridArtifactsPeriodXInMeters.getValue(),
            remove_grid_artifacts_period_y_m=self._objectSettings.removeGridArtifactsPeriodYInMeters.getValue(),
            remove_grid_artifacts_window_size=self._objectSettings.removeGridArtifactsWindowSizeInPixels.getValue(),
            remove_grid_artifacts_direction=remove_grid_artifacts_direction,
            remove_grid_artifacts_stride=self._objectSettings.removeGridArtifactsStride.getValue(),
            multislice_regularization_weight=multislice_regularization_weight,
            multislice_regularization_unwrap_phase=self._objectSettings.regularizeMultisliceUnwrapPhase.getValue(),
            multislice_regularization_unwrap_image_grad_method=multislice_regularization_unwrap_image_grad_method,
            multislice_regularization_stride=self._objectSettings.regularizeMultisliceStride.getValue(),
            patch_interpolation_method=patch_interpolation_method,
        )

    def _create_probe_options(self, probe: Probe) -> DMProbeOptions:
        optimization_plan = self._create_optimization_plan(
            self._probeSettings.optimizationPlanStart.getValue(),
            self._probeSettings.optimizationPlanStop.getValue(),
            self._probeSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._probeSettings.optimizer.getValue())

        probe_power = (
            self._probeSettings.probePower.getValue()
            if self._probeSettings.constrainProbePower.getValue()
            else 0.0
        )

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

        return DMProbeOptions(
            optimizable=self._probeSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._probeSettings.stepSize.getValue(),
            initial_guess=probe.getArray(),
            probe_power=probe_power,
            probe_power_constraint_stride=self._probeSettings.constrainProbePowerStride.getValue(),
            orthogonalize_incoherent_modes=self._probeSettings.orthogonalizeIncoherentModes.getValue(),
            orthogonalize_incoherent_modes_method=orthogonalize_incoherent_modes_method,
            orthogonalize_incoherent_modes_stride=self._probeSettings.orthogonalizeIncoherentModesStride.getValue(),
            orthogonalize_opr_modes=self._probeSettings.orthogonalizeOPRModes.getValue(),
            orthogonalize_opr_modes_stride=self._probeSettings.orthogonalizeOPRModesStride.getValue(),
        )

    def _create_probe_position_options(
        self, scan: Scan, object_geometry: ObjectGeometry
    ) -> DMProbePositionOptions:
        position_in_px_list: list[float] = list()

        for scan_point in scan:
            # FIXME object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            # FIXME position_in_px_list.append(object_point.positionYInPixels)
            # FIXME position_in_px_list.append(object_point.positionXInPixels)
            position_in_px_list.append(
                scan_point.positionXInMeters / object_geometry.pixelWidthInMeters
            )
            position_in_px_list.append(
                scan_point.positionYInMeters / object_geometry.pixelHeightInMeters
            )

        position_in_px = numpy.reshape(position_in_px_list, shape=(len(scan), 2))
        position_in_px -= position_in_px.mean(axis=0)  # FIXME

        probe_position_optimization_plan = self._create_optimization_plan(
            self._probePositionSettings.optimizationPlanStart.getValue(),
            self._probePositionSettings.optimizationPlanStop.getValue(),
            self._probePositionSettings.optimizationPlanStride.getValue(),
        )
        probe_position_optimizer = self._create_optimizer(
            self._probePositionSettings.optimizer.getValue()
        )
        update_magnitude_limit = (
            self._probePositionSettings.magnitudeUpdateLimit.getValue()
            if self._probePositionSettings.limitMagnitudeUpdate.getValue()
            else 0.0
        )

        return DMProbePositionOptions(
            optimizable=self._probePositionSettings.isOptimizable.getValue(),
            optimization_plan=probe_position_optimization_plan,
            optimizer=probe_position_optimizer,
            step_size=self._probePositionSettings.stepSize.getValue(),
            position_x_px=position_in_px[:, -1],
            position_y_px=position_in_px[:, -2],
            update_magnitude_limit=update_magnitude_limit if update_magnitude_limit > 0.0 else None,
            constrain_position_mean=self._probePositionSettings.constrainCentroid.getValue(),
        )

    def _create_opr_mode_weight_options(self) -> DMOPRModeWeightsOptions:
        opr_optimization_plan = self._create_optimization_plan(
            self._oprSettings.optimizationPlanStart.getValue(),
            self._oprSettings.optimizationPlanStop.getValue(),
            self._oprSettings.optimizationPlanStride.getValue(),
        )
        opr_optimizer = self._create_optimizer(self._oprSettings.optimizer.getValue())
        return DMOPRModeWeightsOptions(
            optimizable=self._oprSettings.isOptimizable.getValue(),
            optimization_plan=opr_optimization_plan,
            optimizer=opr_optimizer,
            step_size=self._oprSettings.stepSize.getValue(),
            initial_weights=numpy.array([0.0]),  # FIXME
            optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
            optimize_intensity_variation=self._oprSettings.optimizeIntensities.getValue(),
        )

    def _create_task_options(self, parameters: ReconstructInput) -> DMOptions:
        product = parameters.product
        return DMOptions(
            data_options=self._create_data_options(
                parameters.patterns, parameters.goodPixelMask, product.metadata
            ),
            reconstructor_options=self._create_reconstructor_options(),
            object_options=self._create_object_options(product.object_),
            probe_options=self._create_probe_options(product.probe),
            probe_position_options=self._create_probe_position_options(
                product.scan, product.object_.getGeometry()
            ),
            opr_mode_weight_options=self._create_opr_mode_weight_options(),
        )

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        task = PtychographyTask(self._create_task_options(parameters))
        # TODO task.iterate(n_epochs)
        task.run()

        position_out_px = task.get_data_to_cpu('probe_positions', as_numpy=True)  # FIXME units?
        probe_out_array = task.get_data_to_cpu('probe', as_numpy=True)
        object_out_array = task.get_data_to_cpu('object', as_numpy=True)
        # TODO opr_mode_weights = task.get_data_to_cpu('opr_mode_weights', as_numpy=True)

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.array(object_out_array),
            layerDistanceInMeters=object_in.layerDistanceInMeters,
            pixelGeometry=object_in.getPixelGeometry(),
            centerXInMeters=object_in.centerXInMeters,
            centerYInMeters=object_in.centerYInMeters,
        )

        probe_in = parameters.product.probe
        probe_out = Probe(
            array=numpy.array(probe_out_array[0]),
            pixelGeometry=probe_in.getPixelGeometry(),
        )

        scan_in = parameters.product.scan
        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.getGeometry()

        for uncorrected_point, xy in zip(scan_in, position_out_px):
            object_point = ObjectPoint(uncorrected_point.index, xy[-1], xy[-2])
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
