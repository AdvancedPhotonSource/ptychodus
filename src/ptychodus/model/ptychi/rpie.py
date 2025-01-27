from collections.abc import Sequence
import logging

import numpy

from ptychi.api import (
    BatchingModes,
    Devices,
    Directions,
    Dtypes,
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
    RPIEOptions,
    RPIEReconstructorOptions,
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
from .helper import PtyChiOptionsHelper
from .settings import (
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiReconstructorSettings,
)

logger = logging.getLogger(__name__)


class RPIEReconstructor(Reconstructor):
    def __init__(
        self,
        options_helper: PtyChiOptionsHelper,
        pieSettings: PtyChiPIESettings,
    ) -> None:
        super().__init__()
        self._options_helper = options_helper
        self._pieSettings = pieSettings

    @property
    def name(self) -> str:
        return 'rPIE'

    def _create_reconstructor_options(self) -> RPIEReconstructorOptions:
        helper = self._options_helper.reconstructor_helper
        return RPIEReconstructorOptions(
            num_epochs=helper.num_epochs,
            batch_size=helper.batch_size,
            batching_mode=helper.batching_mode,
            compact_mode_update_clustering=helper.compact_mode_update_clustering,
            compact_mode_update_clustering_stride=helper.compact_mode_update_clustering_stride,
            default_device=helper.default_device,
            default_dtype=helper.default_dtype,
            random_seed=helper.random_seed,
            displayed_loss_function=helper.displayed_loss_function,
            use_low_memory_forward_model=helper.use_low_memory_forward_model,
        )

    def _create_object_options(self, object_: Object) -> PIEObjectOptions:
        helper = self._options_helper.object_helper
        return PIEObjectOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_guess=helper.get_initial_guess(object_),
            slice_spacings_m=helper.get_slice_spacings_m(object_),
            pixel_size_m=helper.get_pixel_size_m(object_),
            l1_norm_constraint=helper.l1_norm_constraint,
            smoothness_constraint=helper.smoothness_constraint,
            total_variation=helper.total_variation,
            remove_grid_artifacts=helper.remove_grid_artifacts,
            multislice_regularization=helper.multislice_regularization,
            patch_interpolation_method=helper.patch_interpolation_method,
            alpha=self._pieSettings.objectAlpha.getValue(),
        )

    def _create_probe_options(self, probe: Probe, metadata: ProductMetadata) -> PIEProbeOptions:
        helper = self._options_helper.probe_helper
        return PIEProbeOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_guess=helper.get_initial_guess(probe),
            power_constraint=helper.get_power_constraint(metadata),
            orthogonalize_incoherent_modes=helper.orthogonalize_incoherent_modes,
            orthogonalize_opr_modes=helper.orthogonalize_opr_modes,
            support_constraint=helper.support_constraint,
            center_constraint=helper.center_constraint,
            eigenmode_update_relaxation=helper.eigenmode_update_relaxation,
            alpha=self._pieSettings.probeAlpha.getValue(),
        )

    def _create_probe_position_options(
        self, scan: Scan, object_geometry: ObjectGeometry
    ) -> PIEProbePositionOptions:
        helper = self._options_helper.probe_position_helper

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

        magnitude_limit_optimization_plan = self._options_helper.create_optimization_plan(
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
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            position_x_px=position_in_px[:, -1],
            position_y_px=position_in_px[:, -2],
            magnitude_limit=magnitude_limit,
            constrain_position_mean=self._probePositionSettings.constrainCentroid.getValue(),
            correction_options=correction_options,
        )

    def _create_opr_mode_weight_options(self) -> PIEOPRModeWeightsOptions:
        helper = self._options_helper.opr_helper

        ####

        smoothing_optimization_plan = self._options_helper.create_optimization_plan(
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
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_weights=numpy.array([0.0]),  # FIXME
            optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
            optimize_intensity_variation=self._oprSettings.optimizeIntensities.getValue(),
            smoothing=smoothing,
            update_relaxation=self._oprSettings.relaxUpdate.getValue(),
        )

    def _create_task_options(self, parameters: ReconstructInput) -> RPIEOptions:
        product = parameters.product
        return RPIEOptions(
            data_options=self._options_helper.create_data_options(parameters),
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
        task.run()  # TODO (n_epochs: int | None = None)

        # TODO rename to position_x_px and position_y_px
        position_out_x_px = task.get_probe_positions_x(as_numpy=True)
        position_out_y_px = task.get_probe_positions_y(as_numpy=True)
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
            array=numpy.array(probe_out_array[0]),  # TODO OPR
            pixelGeometry=probe_in.getPixelGeometry(),
        )

        scan_in = parameters.product.scan
        corrected_scan_points: list[ScanPoint] = list()
        object_geometry = object_in.getGeometry()
        rx_px = object_geometry.widthInPixels / 2
        ry_px = object_geometry.heightInPixels / 2

        for uncorrected_point, pos_x_px, pos_y_px in zip(
            scan_in, position_out_x_px, position_out_y_px
        ):
            object_point = ObjectPoint(
                index=uncorrected_point.index,
                positionXInPixels=pos_x_px + rx_px,
                positionYInPixels=pos_y_px + ry_px,
            )
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = Scan(corrected_scan_points)

        costs: Sequence[float] = list()
        task_reconstructor = task.reconstructor

        if task_reconstructor is not None:
            loss_tracker = task_reconstructor.loss_tracker
            # TODO update api to include epoch and loss
            # epoch = loss_tracker.table['epoch'].to_numpy()
            loss = loss_tracker.table['loss'].to_numpy()
            costs = [float(x) for x in loss.flatten()]

        product = Product(
            metadata=parameters.product.metadata,
            scan=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )

        return ReconstructOutput(product, 0)
