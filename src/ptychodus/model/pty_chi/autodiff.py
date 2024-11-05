import logging

import numpy

from ptychi.api import (
    AutodiffPtychographyOPRModeWeightsOptions,
    AutodiffPtychographyObjectOptions,
    AutodiffPtychographyOptions,
    AutodiffPtychographyProbeOptions,
    AutodiffPtychographyProbePositionOptions,
    AutodiffPtychographyReconstructorOptions,
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
    ImageGradientMethods,
    OptimizationPlan,
    Optimizers,
    OrthogonalizationMethods,
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

# FEEDBACK
# change = None options to required arguments, rest have default values
# remove positions/object pixel sizes
# switch parameters with strides to "correction plans"
# remove orthogonalize_incoherent_modes: enable whenever stride >= 1
# remove orthogonalize_opr_modes: enable whenever stride >= 1
# remove num_epochs from reconstructor options and run() method; just use iterate(n_epochs)
# remove log_level; just document that logging is used in a standard way
# better name for l1_norm_constraint?
# enum for get_data_to_cpu
# expand optimization mode GS; stabilize?
# remove update_magnitude_limit?
# remove optimizable in favor of OptimizationPlan | None
# add optimizer and step size to OptimizationPlan
# add additional enum value to avoid optional enum
# return loss function values


class AutodiffReconstructor(Reconstructor):
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
        return 'Autodiff'

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

    def _create_reconstructor_options(self) -> AutodiffPtychographyReconstructorOptions:
        batching_mode_str = self._reconstructorSettings.batchingMode.getValue()

        try:
            batching_mode = BatchingModes[batching_mode_str.upper()]
        except KeyError:
            logger.warning('Failed to parse batching mode "{batching_mode_str}"!')
            batching_mode = BatchingModes.RANDOM

        return AutodiffPtychographyReconstructorOptions(
            num_epochs=self._reconstructorSettings.numEpochs.getValue(),
            batch_size=self._reconstructorSettings.batchSize.getValue(),
            batching_mode=batching_mode,
            compact_mode_update_clustering=self._reconstructorSettings.compactModeUpdateClustering.getValue(),
            compact_mode_update_clustering_stride=self._reconstructorSettings.compactModeUpdateClusteringStride.getValue(),
            # TODO default_device=Devices.GPU if self._reconstructorSettings.useDevices.getValue() else Devices.CPU,
            # TODO gpu_indices=self._reconstructorSettings.devices.getValue(),
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

    def _create_object_options(self, object_: Object) -> AutodiffPtychographyObjectOptions:
        optimization_plan = self._create_optimization_plan(
            self._objectSettings.optimizationPlanStart.getValue(),
            self._objectSettings.optimizationPlanStop.getValue(),
            self._objectSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._objectSettings.optimizer.getValue())

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

        multislice_regularization_unwrap_image_grad_method_str = (
            self._objectSettings.multisliceRegularizationUnwrapImageGradMethod.getValue()
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

        return AutodiffPtychographyObjectOptions(
            optimizable=self._objectSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._objectSettings.stepSize.getValue(),
            initial_guess=object_.array,
            slice_spacings_m=numpy.array(object_.layerDistanceInMeters[:-1]),
            l1_norm_constraint_weight=self._objectSettings.l1NormConstraintWeight.getValue(),
            l1_norm_constraint_stride=self._objectSettings.l1NormConstraintStride.getValue(),
            smoothness_constraint_alpha=self._objectSettings.smoothnessConstraintAlpha.getValue(),
            smoothness_constraint_stride=self._objectSettings.smoothnessConstraintStride.getValue(),
            total_variation_weight=self._objectSettings.totalVariationWeight.getValue(),
            total_variation_stride=self._objectSettings.totalVaritionStride.getValue(),
            remove_grid_artifacts=self._objectSettings.removeGridArtifacts.getValue(),
            remove_grid_artifacts_period_x_m=self._objectSettings.removeGridArtifactsPeriodXInMeters.getValue(),
            remove_grid_artifacts_period_y_m=self._objectSettings.removeGridArtifactsPeriodYInMeters.getValue(),
            remove_grid_artifacts_window_size=self._objectSettings.removeGridArtifactsWindowSizeInPixels.getValue(),
            remove_grid_artifacts_direction=remove_grid_artifacts_direction,
            remove_grid_artifacts_stride=self._objectSettings.removeGridArtifactsStride.getValue(),
            multislice_regularization_weight=self._objectSettings.multisliceRegularizationWeight.getValue(),
            multislice_regularization_unwrap_phase=self._objectSettings.multisliceRegularizationUnwrapPhase.getValue(),
            multislice_regularization_unwrap_image_grad_method=multislice_regularization_unwrap_image_grad_method,
            multislice_regularization_stride=self._objectSettings.multisliceRegularizationStride.getValue(),
        )

    def _create_probe_options(self, probe: Probe) -> AutodiffPtychographyProbeOptions:
        optimization_plan = self._create_optimization_plan(
            self._probeSettings.optimizationPlanStart.getValue(),
            self._probeSettings.optimizationPlanStop.getValue(),
            self._probeSettings.optimizationPlanStride.getValue(),
        )
        optimizer = self._create_optimizer(self._probeSettings.optimizer.getValue())
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

        return AutodiffPtychographyProbeOptions(
            optimizable=self._probeSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=optimization_plan,
            optimizer=optimizer,
            step_size=self._probeSettings.stepSize.getValue(),
            initial_guess=probe.array[numpy.newaxis, ...],  # TODO opr
            probe_power=self._probeSettings.probePower.getValue(),
            probe_power_constraint_stride=self._probeSettings.probePowerConstraintStride.getValue(),
            orthogonalize_incoherent_modes=self._probeSettings.orthogonalizeIncoherentModes.getValue(),
            orthogonalize_incoherent_modes_stride=self._probeSettings.orthogonalizeIncoherentModesStride.getValue(),
            orthogonalize_incoherent_modes_method=orthogonalize_incoherent_modes_method,
            orthogonalize_opr_modes=self._probeSettings.orthogonalizeOPRModes.getValue(),
            orthogonalize_opr_modes_stride=self._probeSettings.orthogonalizeOPRModesStride.getValue(),
        )

    def _create_probe_position_options(
        self, scan: Scan, object_geometry: ObjectGeometry
    ) -> AutodiffPtychographyProbePositionOptions:
        position_in_px_list: list[float] = list()

        for scan_point in scan:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            position_in_px_list.append(object_point.positionYInPixels)
            position_in_px_list.append(object_point.positionXInPixels)

        position_in_px = numpy.reshape(position_in_px_list, shape=(len(scan), 2))

        probe_position_optimization_plan = self._create_optimization_plan(
            self._probePositionSettings.optimizationPlanStart.getValue(),
            self._probePositionSettings.optimizationPlanStop.getValue(),
            self._probePositionSettings.optimizationPlanStride.getValue(),
        )
        probe_position_optimizer = self._create_optimizer(
            self._probePositionSettings.optimizer.getValue()
        )
        update_magnitude_limit = self._probePositionSettings.updateMagnitudeLimit.getValue()

        return AutodiffPtychographyProbePositionOptions(
            optimizable=self._probePositionSettings.isOptimizable.getValue(),
            optimization_plan=probe_position_optimization_plan,
            optimizer=probe_position_optimizer,
            step_size=self._probePositionSettings.stepSize.getValue(),
            position_x_px=position_in_px[:, -1],
            position_y_px=position_in_px[:, -2],
            update_magnitude_limit=update_magnitude_limit if update_magnitude_limit > 0.0 else None,
            constrain_position_mean=self._probePositionSettings.constrainPositionMean.getValue(),
        )

    def _create_opr_mode_weight_options(self) -> AutodiffPtychographyOPRModeWeightsOptions:
        opr_optimization_plan = self._create_optimization_plan(
            self._oprSettings.optimizationPlanStart.getValue(),
            self._oprSettings.optimizationPlanStop.getValue(),
            self._oprSettings.optimizationPlanStride.getValue(),
        )
        opr_optimizer = self._create_optimizer(self._oprSettings.optimizer.getValue())
        return AutodiffPtychographyOPRModeWeightsOptions(
            optimizable=self._oprSettings.isOptimizable.getValue(),
            optimization_plan=opr_optimization_plan,
            optimizer=opr_optimizer,
            step_size=self._oprSettings.stepSize.getValue(),
            initial_weights=numpy.array([0.0]),  # FIXME
            optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
            optimize_intensity_variation=self._oprSettings.optimizeIntensities.getValue(),
        )

    def _create_task_options(self, parameters: ReconstructInput) -> AutodiffPtychographyOptions:
        product = parameters.product
        return AutodiffPtychographyOptions(
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
            layerDistanceInMeters=object_in.layerDistanceInMeters,  # TODO verify optimized?
            pixelWidthInMeters=object_in.pixelWidthInMeters,
            pixelHeightInMeters=object_in.pixelHeightInMeters,
            centerXInMeters=object_in.centerXInMeters,
            centerYInMeters=object_in.centerYInMeters,
        )

        probe_in = parameters.product.probe
        probe_out = Probe(
            array=numpy.array(probe_out_array),
            pixelWidthInMeters=probe_in.pixelWidthInMeters,
            pixelHeightInMeters=probe_in.pixelHeightInMeters,
        )

        scan_in = parameters.product.scan
        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.getGeometry()

        for uncorrected_point, xy in zip(scan_in, position_out_px):
            object_point = ObjectPoint(uncorrected_point.index, xy[-1], xy[-2])
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = Scan(corrected_scan_points)

        costs: list[float] = list()  # TODO populate

        product = Product(
            metadata=parameters.product.metadata,
            scan=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )

        return ReconstructOutput(product, 0)
