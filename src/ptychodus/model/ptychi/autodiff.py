import logging


from ptychi.api import (
    AutodiffPtychographyOPRModeWeightsOptions,
    AutodiffPtychographyObjectOptions,
    AutodiffPtychographyOptions,
    AutodiffPtychographyProbeOptions,
    AutodiffPtychographyProbePositionOptions,
    AutodiffPtychographyReconstructorOptions,
    ForwardModels,
    LossFunctions,
)
from ptychi.api.task import PtychographyTask

from ptychodus.api.object import Object, ObjectGeometry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, ProductMetadata
from ptychodus.api.reconstructor import ReconstructInput, ReconstructOutput, Reconstructor
from ptychodus.api.scan import PositionSequence

from .helper import PtyChiOptionsHelper
from .settings import PtyChiAutodiffSettings

logger = logging.getLogger(__name__)


class AutodiffReconstructor(Reconstructor):
    def __init__(
        self, options_helper: PtyChiOptionsHelper, settings: PtyChiAutodiffSettings
    ) -> None:
        super().__init__()
        self._options_helper = options_helper
        self._settings = settings

    @property
    def name(self) -> str:
        return 'Autodiff'

    def _create_reconstructor_options(self) -> AutodiffPtychographyReconstructorOptions:
        helper = self._options_helper.reconstructor_helper

        ####

        loss_function_str = self._settings.loss_function.get_value()

        try:
            loss_function = LossFunctions[loss_function_str.upper()]
        except KeyError:
            logger.warning('Failed to parse loss function "{loss_function_str}"!')
            loss_function = LossFunctions.MSE_SQRT

        ####

        forward_model_class_str = self._settings.forward_model_class.get_value()

        try:
            forward_model_class = ForwardModels[forward_model_class_str.upper()]
        except KeyError:
            logger.warning('Failed to parse forward model class "{forward_model_class_str}"!')
            forward_model_class = ForwardModels.PLANAR_PTYCHOGRAPHY

        ####

        return AutodiffPtychographyReconstructorOptions(
            num_epochs=helper.num_epochs,
            batch_size=helper.batch_size,
            batching_mode=helper.batching_mode,
            compact_mode_update_clustering=helper.compact_mode_update_clustering,
            compact_mode_update_clustering_stride=helper.compact_mode_update_clustering_stride,
            default_device=helper.default_device,
            default_dtype=helper.default_dtype,
            use_double_precision_for_fft=helper.use_double_precision_for_fft,
            allow_nondeterministic_algorithms=helper.allow_nondeterministic_algorithms,
            random_seed=helper.random_seed,
            displayed_loss_function=helper.displayed_loss_function,
            forward_model_options=helper.forward_model_options,
            loss_function=loss_function,
            forward_model_class=forward_model_class,
            forward_model_params=None,
        )

    def _create_object_options(self, object_: Object) -> AutodiffPtychographyObjectOptions:
        helper = self._options_helper.object_helper
        return AutodiffPtychographyObjectOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_guess=helper.get_initial_guess(object_),
            slice_spacings_m=helper.get_slice_spacings_m(object_),
            slice_spacing_options=helper.slice_spacing_options,
            pixel_size_m=helper.get_pixel_size_m(object_),
            pixel_size_aspect_ratio=helper.get_pixel_aspect_ratio(object_),
            l1_norm_constraint=helper.l1_norm_constraint,
            l2_norm_constraint=helper.l2_norm_constraint,
            smoothness_constraint=helper.smoothness_constraint,
            total_variation=helper.total_variation,
            remove_grid_artifacts=helper.remove_grid_artifacts,
            multislice_regularization=helper.multislice_regularization,
            patch_interpolation_method=helper.patch_interpolation_method,
            remove_object_probe_ambiguity=helper.remove_object_probe_ambiguity,
            build_preconditioner_with_all_modes=helper.build_preconditioner_with_all_modes,
            determine_position_origin_coords_by=helper.determine_position_origin_coords_by,
            position_origin_coords=helper.get_position_origin_coords(object_),
        )

    def _create_probe_options(
        self, probes: ProbeSequence, metadata: ProductMetadata
    ) -> AutodiffPtychographyProbeOptions:
        helper = self._options_helper.probe_helper
        return AutodiffPtychographyProbeOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_guess=helper.get_initial_guess(probes),
            power_constraint=helper.get_power_constraint(metadata),
            orthogonalize_incoherent_modes=helper.orthogonalize_incoherent_modes,
            orthogonalize_opr_modes=helper.orthogonalize_opr_modes,
            support_constraint=helper.support_constraint,
            center_constraint=helper.center_constraint,
            eigenmode_update_relaxation=helper.eigenmode_update_relaxation,
        )

    def _create_probe_position_options(
        self, scan: PositionSequence, object_geometry: ObjectGeometry
    ) -> AutodiffPtychographyProbePositionOptions:
        helper = self._options_helper.probe_position_helper
        position_x_px, position_y_px = helper.get_positions_px(scan, object_geometry)
        return AutodiffPtychographyProbePositionOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            position_x_px=position_x_px,
            position_y_px=position_y_px,
            constrain_position_mean=helper.constrain_position_mean,
            correction_options=helper.correction_options,
            affine_transform_constraint=helper.affine_transform_constraint,
        )

    def _create_opr_mode_weight_options(
        self, probes: ProbeSequence
    ) -> AutodiffPtychographyOPRModeWeightsOptions:
        helper = self._options_helper.opr_helper
        return AutodiffPtychographyOPRModeWeightsOptions(
            optimizable=helper.optimizable,
            optimization_plan=helper.optimization_plan,
            optimizer=helper.optimizer,
            step_size=helper.step_size,
            optimizer_params=helper.optimizer_params,
            initial_weights=helper.get_initial_weights(probes),
            optimize_eigenmode_weights=helper.optimize_eigenmode_weights,
            optimize_intensity_variation=helper.optimize_intensity_variation,
            smoothing=helper.smoothing,
            update_relaxation=helper.update_relaxation,
        )

    def _create_task_options(self, parameters: ReconstructInput) -> AutodiffPtychographyOptions:
        product = parameters.product
        return AutodiffPtychographyOptions(
            data_options=self._options_helper.create_data_options(parameters),
            reconstructor_options=self._create_reconstructor_options(),
            object_options=self._create_object_options(product.object_),
            probe_options=self._create_probe_options(product.probes, product.metadata),
            probe_position_options=self._create_probe_position_options(
                product.positions, product.object_.get_geometry()
            ),
            opr_mode_weight_options=self._create_opr_mode_weight_options(product.probes),
        )

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        task_options = self._create_task_options(parameters)
        task = PtychographyTask(task_options)
        task.run()  # TODO (n_epochs: int | None = None)

        losses: list[LossValue] = list()
        task_reconstructor = task.reconstructor

        if task_reconstructor is not None:
            loss_tracker = task_reconstructor.loss_tracker
            epoch_array = loss_tracker.table['epoch'].to_numpy()
            loss_array = loss_tracker.table['loss'].to_numpy()

            for epoch, loss in zip(epoch_array.flat, loss_array.flat):
                losses.append(LossValue(epoch=epoch, value=loss.item()))

        product = self._options_helper.create_product(
            product=parameters.product,
            position_x_px=task.get_probe_positions_x(as_numpy=True),
            position_y_px=task.get_probe_positions_y(as_numpy=True),
            probe_array=task.get_data_to_cpu('probe', as_numpy=True),
            object_array=task.get_data_to_cpu('object', as_numpy=True),
            opr_weights=task.get_data_to_cpu('opr_mode_weights', as_numpy=True),
            losses=losses,
        )
        return ReconstructOutput(product, 0)
