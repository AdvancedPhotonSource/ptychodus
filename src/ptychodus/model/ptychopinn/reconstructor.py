from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final
import logging

import numpy
import numpy.typing

from ptycho.config.config import InferenceConfig, ModelConfig, TrainingConfig, update_legacy_dict
from ptycho.raw_data import RawData
import ptycho.loader
import ptycho.model_manager
import ptycho.params

from ptychodus.api.object import Object
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    LossValue,
    ReconstructInput,
    ReconstructOutput,
    TrainOutput,
    TrainableReconstructor,
)

from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

__all__ = [
    'PtychoPINNTrainableReconstructor',
]

logger = logging.getLogger(__name__)


def create_raw_data(parameters: ReconstructInput) -> RawData:
    object_geometry = parameters.product.object_.get_geometry()
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in parameters.product.positions:
        object_point = object_geometry.map_scan_point_to_object_point(scan_point)
        position_x_px.append(object_point.position_x_px)
        position_y_px.append(object_point.position_y_px)

    return RawData.from_coords_without_pc(
        xcoords=numpy.array(position_x_px),
        ycoords=numpy.array(position_y_px),
        diff3d=parameters.patterns,
        probeGuess=parameters.product.probes.get_probe_no_opr().get_incoherent_mode(0),
        # assume that all patches are from the same object
        scan_index=numpy.zeros(len(parameters.product.positions), dtype=int),
        objectGuess=parameters.product.object_.get_layer(0),
    )


class PtychoPINNTrainableReconstructor(TrainableReconstructor):
    MODEL_FILE_FILTER: Final[str] = 'Zipped Archive (*.zip)'
    TRAINING_DATA_FILE_FILTER: Final[str] = 'NumPy Zipped Archive (*.npz)'

    # TODO datasets for testing: xpp, "u", ALS
    # TODO normalize data in preprocessing step (see note in slack)
    # TODO ptychodus stitches

    def __init__(
        self,
        name: str,
        model_settings: PtychoPINNModelSettings,
        inference_settings: PtychoPINNInferenceSettings,
        training_settings: PtychoPINNTrainingSettings,
        *in_developer_mode: bool,
    ) -> None:
        super().__init__()
        self._name = name
        self._model_settings = model_settings
        self._inference_settings = inference_settings
        self._training_settings = training_settings
        self._model_dict: dict[str, Any] | None = None
        self._in_developer_mode = in_developer_mode

        ptychopinn_version = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinn_version}')

    def _create_model_config(self, model_size: int) -> ModelConfig:
        return ModelConfig(
            N=model_size,
            gridsize=self._model_settings.gridsize.get_value(),
            n_filters_scale=self._model_settings.n_filters_scale.get_value(),
            model_type=self._name.lower(),
            amp_activation=self._model_settings.amp_activation.get_value(),
            object_big=self._model_settings.object_big.get_value(),
            probe_big=self._model_settings.probe_big.get_value(),
            probe_mask=self._model_settings.probe_mask.get_value(),
            pad_object=self._model_settings.pad_object.get_value(),
            probe_scale=self._model_settings.probe_scale.get_value(),
            gaussian_smoothing_sigma=self._model_settings.gaussian_smoothing_sigma.get_value(),
        )

    @property
    def name(self) -> str:
        return self._name

    def _reconstruct_image(self, test_data: ptycho.loader.PtychoDataContainer) -> Any:
        if self._model_dict is None:
            raise RuntimeError('Model not loaded!')

        import ptycho.model

        diffraction_to_obj = self._model_dict['diffraction_to_obj']  # tf.keras.Model
        intensity_scale = ptycho.model.params()['intensity_scale']
        return diffraction_to_obj.predict([test_data.X * intensity_scale, test_data.local_offsets])

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        model_size = parameters.patterns.shape[-1]

        if parameters.patterns.shape[-2] != model_size:
            raise ValueError('Model requires square diffraction patterns!')

        if self._model_dict is None:
            raise ValueError('Model not loaded.')

        model_config = self._create_model_config(model_size)
        inference_config = InferenceConfig(
            model=model_config,
            model_path=Path(),  # not used
            test_data_file=Path(),  # not used
            debug=self._in_developer_mode,
            output_dir=Path(),  # not used
        )

        # Update global params with new-style config
        update_legacy_dict(ptycho.params.cfg, inference_config)

        # Create RawData
        test_raw_data = create_raw_data(parameters)
        ptycho.probe.set_probe_guess(None, test_raw_data.probeGuess)

        # Group overlapping scan positions
        test_dataset = test_raw_data.generate_grouped_data(
            model_config.N,
            K=self._inference_settings.n_nearest_neighbors.get_value(),
            nsamples=self._inference_settings.n_samples.get_value(),
        )

        # Create PtychoDataContainer
        test_data_container = ptycho.loader.load(
            lambda: test_dataset, test_raw_data.probeGuess, which=None, create_split=False
        )

        # Perform reconstruction
        obj_tensor_full = self._reconstruct_image(test_data_container)

        # Process the reconstructed image
        object_out_array = ptycho.tf_helper.reassemble_position(
            obj_tensor_full, test_data_container.global_offsets, M=20
        )

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.squeeze(object_out_array),
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )
        costs: Sequence[float] = list()

        product = Product(
            metadata=parameters.product.metadata,
            positions=parameters.product.positions,
            probes=parameters.product.probes,
            object_=object_out,
            costs=costs,
        )

        return ReconstructOutput(product, 0)

    def get_model_file_filter(self) -> str:
        return self.MODEL_FILE_FILTER

    def open_model(self, file_path: Path) -> None:
        # TODO model path to/from settings
        self._inference_settings.model_path.set_value(file_path)
        # ModelManager updates global config (ptycho.params.cfg) when loading
        self._model_dict = ptycho.model_manager.ModelManager.load_multiple_models(
            file_path.parent / file_path.stem
        )
        # TODO update settings from ptycho.params.cfg after loading

    def save_model(self, file_path: Path) -> None:
        ptycho.model_manager.save(file_path)

    def get_training_data_file_filter(self) -> str:
        return self.TRAINING_DATA_FILE_FILTER

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        object_geometry = parameters.product.object_.get_geometry()
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()

        for scan_point in parameters.product.positions:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            position_x_px.append(object_point.position_x_px)
            position_y_px.append(object_point.position_y_px)

        xcoords = numpy.array(position_x_px)
        ycoords = numpy.array(position_y_px)

        numpy.savez(
            file_path,
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=parameters.patterns,
            probeGuess=parameters.product.probes.get_probe_no_opr().get_incoherent_mode(0),
            # assume that all patches are from the same object
            objectGuess=parameters.product.object_.get_layer(0),
            scan_index=numpy.zeros(len(parameters.product.positions), dtype=int),
        )

    def get_training_data_path(self) -> Path:
        return self._training_settings.data_dir.get_value()

    def train(self, data_path: Path) -> TrainOutput:
        self._training_settings.data_dir.set_value(data_path)

        test_raw_data = RawData.from_file(data_path / 'test_data.npz')  # TODO RawData | None
        train_raw_data = RawData.from_file(data_path / 'train_data.npz')

        model_size = train_raw_data.diff3d.shape[-1]

        if train_raw_data.diff3d.shape[-2] != model_size:
            raise ValueError('Model requires square diffraction patterns!')

        model_config = self._create_model_config(model_size)
        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path(),  # not used
            test_data_file=None,  # not used
            batch_size=self._training_settings.batch_size.get_value(),
            nepochs=self._training_settings.nepochs.get_value(),
            mae_weight=self._training_settings.mae_weight.get_value(),
            nll_weight=self._training_settings.nll_weight.get_value(),
            realspace_mae_weight=self._training_settings.realspace_mae_weight.get_value(),
            realspace_weight=self._training_settings.realspace_weight.get_value(),
            nphotons=self._training_settings.nphotons.get_value(),  # TODO get from product
            positions_provided=self._training_settings.positions_provided.get_value(),
            probe_trainable=self._training_settings.probe_trainable.get_value(),
            intensity_scale_trainable=self._training_settings.intensity_scale_trainable.get_value(),
            output_dir=Path(),  # not used
        )

        # Update global params with new-style config
        update_legacy_dict(ptycho.params.cfg, training_config)

        from ptycho.workflows.components import run_cdi_example, save_outputs

        recon_amp, recon_phase, train_results = run_cdi_example(
            train_raw_data, test_raw_data, training_config
        )
        output_dir = self._training_settings.output_dir.get_value()
        self.save_model(output_dir)
        save_outputs(recon_amp, recon_phase, train_results, str(output_dir))
        print(train_results.keys())  # TODO remove
        # dict_keys(['history', 'model_instance', 'reconstructed_obj', 'pred_amp', 'reconstructed_obj_cdi', 'stitched_obj', 'train_container', 'test_container', 'obj_tensor_full', 'global_offsets', 'recon_amp', 'recon_phase'])
        # TODO self._model_dict = train_results

        losses: Sequence[LossValue] = []
        return TrainOutput(losses, 0)  # TODO
