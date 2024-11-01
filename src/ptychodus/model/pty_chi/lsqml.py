import logging

import numpy

from ptychi.api import (
    Devices,
    Dtypes,
    LSQMLOPRModeWeightsOptions,
    LSQMLObjectOptions,
    LSQMLOptions,
    LSQMLProbeOptions,
    LSQMLProbePositionOptions,
    LSQMLReconstructorOptions,
    OptimizationPlan,
    Optimizers,
    OrthogonalizationMethods,
    PtychographyDataOptions,
)
from ptychi.api.task import PtychographyTask

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product
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


class LSQMLReconstructor(Reconstructor):
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
        return 'LSQML'

    def _create_optimization_plan(self, start: int, stop: int, stride: int) -> OptimizationPlan:
        return OptimizationPlan(start, None if stop < 0 else stop, stride)

    def _create_optimizer(self, text: str) -> Optimizers:
        try:
            optimizer = Optimizers[text.upper()]
        except KeyError:
            logger.warning('Failed to parse optimizer "{text}"!')
            optimizer = Optimizers.SGD

        return optimizer

    def _create_orthogonalization_method(self, text: str) -> OrthogonalizationMethods:
        try:
            method = OrthogonalizationMethods[text.upper()]
        except KeyError:
            logger.warning('Failed to parse orthogonalization method "{text}"!')
            method = OrthogonalizationMethods.GS

        return method

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        metadata = parameters.product.metadata
        object_in = parameters.product.object_
        object_geometry = object_in.getGeometry()
        probe_in = parameters.product.probe
        pixel_size_m = parameters.product.object_.pixelWidthInMeters
        scan_in = parameters.product.scan
        position_in_px_list: list[float] = list()

        for scan_point in scan_in:
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            position_in_px_list.append(object_point.positionYInPixels)
            position_in_px_list.append(object_point.positionXInPixels)

        position_in_px = numpy.reshape(position_in_px_list, shape=(len(scan_in), 2))

        data_options = PtychographyDataOptions(
            data=parameters.patterns,
            propagation_distance_m=metadata.detectorDistanceInMeters,
            wavelength_m=metadata.probeWavelengthInMeters,
            detector_pixel_size_m=self._detector.pixelWidthInMeters.getValue(),
            valid_pixel_mask=parameters.goodPixelMask,
        )
        default_device = (
            Devices.GPU if self._reconstructorSettings.useDevices.getValue() else Devices.CPU
        )
        default_dtype = (
            Dtypes.FLOAT64
            if self._reconstructorSettings.useDoublePrecision.getValue()
            else Dtypes.FLOAT32
        )
        reconstructor_options = LSQMLReconstructorOptions(
            num_epochs=self._reconstructorSettings.numEpochs.getValue(),
            batch_size=self._reconstructorSettings.batchSize.getValue(),
            # TODO batching_mode
            # TODO compact_mode_update_clustering
            # TODO compact_mode_update_clustering_stride
            default_device=default_device,
            gpu_indices=self._reconstructorSettings.devices.getValue(),
            default_dtype=default_dtype,
            # TODO random_seed
            # TODO displayed_loss_function
            # TODO log_level
        )
        object_optimization_plan = self._create_optimization_plan(
            self._objectSettings.optimizationPlanStart.getValue(),
            self._objectSettings.optimizationPlanStop.getValue(),
            self._objectSettings.optimizationPlanStride.getValue(),
        )
        object_optimizer = self._create_optimizer(self._objectSettings.optimizer.getValue())
        object_options = LSQMLObjectOptions(
            optimizable=self._objectSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=object_optimization_plan,
            optimizer=object_optimizer,
            step_size=self._objectSettings.stepSize.getValue(),
            initial_guess=object_in.array,
            slice_spacings_m=None,  # TODO Optional[ndarray]
            pixel_size_m=pixel_size_m,
            l1_norm_constraint_weight=self._objectSettings.l1NormConstraintWeight.getValue(),
            l1_norm_constraint_stride=self._objectSettings.l1NormConstraintStride.getValue(),
            smoothness_constraint_alpha=self._objectSettings.smoothnessConstraintAlpha.getValue(),
            smoothness_constraint_stride=self._objectSettings.smoothnessConstraintStride.getValue(),
            total_variation_weight=self._objectSettings.totalVariationWeight.getValue(),
            total_variation_stride=self._objectSettings.totalVaritionStride.getValue(),
        )
        probe_optimization_plan = self._create_optimization_plan(
            self._probeSettings.optimizationPlanStart.getValue(),
            self._probeSettings.optimizationPlanStop.getValue(),
            self._probeSettings.optimizationPlanStride.getValue(),
        )
        probe_optimizer = self._create_optimizer(self._probeSettings.optimizer.getValue())
        probe_orthogonalize_incoherent_modes_method = self._create_orthogonalization_method(
            self._probeSettings.orthogonalizeIncoherentModesMethod.getValue()
        )
        probe_options = LSQMLProbeOptions(
            optimizable=self._probeSettings.isOptimizable.getValue(),  # TODO optimizer_params
            optimization_plan=probe_optimization_plan,
            optimizer=probe_optimizer,
            step_size=self._probeSettings.stepSize.getValue(),
            initial_guess=probe_in.array,
            probe_power=self._probeSettings.probePower.getValue(),
            probe_power_constraint_stride=self._probeSettings.probePowerConstraintStride.getValue(),
            orthogonalize_incoherent_modes=self._probeSettings.orthogonalizeIncoherentModes.getValue(),
            orthogonalize_incoherent_modes_stride=self._probeSettings.orthogonalizeIncoherentModesStride.getValue(),
            orthogonalize_incoherent_modes_method=probe_orthogonalize_incoherent_modes_method,
            orthogonalize_opr_modes=self._probeSettings.orthogonalizeOPRModes.getValue(),
            orthogonalize_opr_modes_stride=self._probeSettings.orthogonalizeOPRModesStride.getValue(),
        )
        probe_position_optimization_plan = self._create_optimization_plan(
            self._probePositionSettings.optimizationPlanStart.getValue(),
            self._probePositionSettings.optimizationPlanStop.getValue(),
            self._probePositionSettings.optimizationPlanStride.getValue(),
        )
        probe_position_optimizer = self._create_optimizer(
            self._probePositionSettings.optimizer.getValue()
        )
        update_magnitude_limit = self._probePositionSettings.updateMagnitudeLimit.getValue()

        probe_position_options = LSQMLProbePositionOptions(
            optimizable=self._probePositionSettings.isOptimizable.getValue(),
            optimization_plan=probe_position_optimization_plan,
            optimizer=probe_position_optimizer,
            step_size=self._probePositionSettings.stepSize.getValue(),
            position_x_px=position_in_px[:, -1],
            position_y_px=position_in_px[:, -2],
            update_magnitude_limit=update_magnitude_limit if update_magnitude_limit > 0.0 else None,
        )
        opr_optimization_plan = self._create_optimization_plan(
            self._oprSettings.optimizationPlanStart.getValue(),
            self._oprSettings.optimizationPlanStop.getValue(),
            self._oprSettings.optimizationPlanStride.getValue(),
        )
        opr_optimizer = self._create_optimizer(self._oprSettings.optimizer.getValue())
        opr_weights = numpy.array([0.0])  # FIXME
        opr_mode_weight_options = LSQMLOPRModeWeightsOptions(
            optimizable=self._oprSettings.isOptimizable.getValue(),
            optimization_plan=opr_optimization_plan,
            optimizer=opr_optimizer,
            step_size=self._oprSettings.stepSize.getValue(),
            initial_weights=opr_weights,
            optimize_eigenmode_weights=self._oprSettings.optimizeEigenmodeWeights.getValue(),
            optimize_intensity_variation=self._oprSettings.optimizeIntensities.getValue(),
        )
        task_options = LSQMLOptions(
            data_options,
            reconstructor_options,
            object_options,
            probe_options,
            probe_position_options,
            opr_mode_weight_options,
        )

        task = PtychographyTask(task_options)
        # TODO task.iterate(n_epochs)
        task.run()

        position_out_px = task.get_data_to_cpu('probe_positions', as_numpy=True)  # FIXME units?
        probe_out_array = task.get_data_to_cpu('probe', as_numpy=True)
        object_out_array = task.get_data_to_cpu('object', as_numpy=True)
        # TODO opr_mode_weights = task.get_data_to_cpu('opr_mode_weights', as_numpy=True)

        corrected_scan_points: list[ScanPoint] = list()

        for uncorrected_point, xy in zip(scan_in, position_out_px):
            object_point = ObjectPoint(uncorrected_point.index, xy[-1], xy[-2])
            scan_point = object_geometry.mapObjectPointToScanPoint(object_point)
            corrected_scan_points.append(scan_point)

        scan_out = Scan(corrected_scan_points)
        probe_out = Probe(
            array=numpy.array(probe_out_array),
            pixelWidthInMeters=probe_in.pixelWidthInMeters,
            pixelHeightInMeters=probe_in.pixelHeightInMeters,
        )
        object_out = Object(
            array=numpy.array(object_out_array),
            layerDistanceInMeters=object_in.layerDistanceInMeters,  # TODO verify optimized?
            pixelWidthInMeters=object_in.pixelWidthInMeters,
            pixelHeightInMeters=object_in.pixelHeightInMeters,
            centerXInMeters=object_in.centerXInMeters,
            centerYInMeters=object_in.centerYInMeters,
        )
        costs: list[float] = list()  # TODO populate

        product = Product(
            metadata=parameters.product.metadata,
            scan=scan_out,
            probe=probe_out,
            object_=object_out,
            costs=costs,
        )
        return ReconstructOutput(product, 0)
