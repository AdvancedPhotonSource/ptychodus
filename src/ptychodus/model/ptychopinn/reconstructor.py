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
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
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
    object_geometry = parameters.product.object_.getGeometry()
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in parameters.product.scan:
        object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
        position_x_px.append(object_point.positionXInPixels)
        position_y_px.append(object_point.positionYInPixels)

    return RawData.from_coords_without_pc(
        xcoords=numpy.array(position_x_px),
        ycoords=numpy.array(position_y_px),
        diff3d=parameters.patterns,
        probeGuess=parameters.product.probe.getIncoherentMode(0),
        # assume that all patches are from the same object
        scan_index=numpy.zeros(len(parameters.product.scan), dtype=int),
        objectGuess=parameters.product.object_.getLayer(0),
    )


class PtychoPINNTrainableReconstructor(TrainableReconstructor):
    MODEL_FILE_FILTER: Final[str] = 'Zipped Archive (*.zip)'
    TRAINING_DATA_FILE_FILTER: Final[str] = 'NumPy Zipped Archive (*.npz)'

    # FIXME datasets for testing: xpp, "u", ALS
    # FIXME normalize data in preprocessing step (see note in slack)
    # FIXME ptychodus stitches

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

        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')

    def _create_model_config(self, model_size: int) -> ModelConfig:
        return ModelConfig(
            N=model_size,
            gridsize=self._model_settings.gridsize.getValue(),
            n_filters_scale=self._model_settings.n_filters_scale.getValue(),
            model_type=self._name.lower(),
            amp_activation=self._model_settings.amp_activation.getValue(),
            object_big=self._model_settings.object_big.getValue(),
            probe_big=self._model_settings.probe_big.getValue(),
            probe_mask=self._model_settings.probe_mask.getValue(),
            pad_object=self._model_settings.pad_object.getValue(),
            probe_scale=self._model_settings.probe_scale.getValue(),
            gaussian_smoothing_sigma=self._model_settings.gaussian_smoothing_sigma.getValue(),
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
            K=self._inference_settings.n_nearest_neighbors.getValue(),
            nsamples=self._inference_settings.n_samples.getValue(),
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
            layerDistanceInMeters=object_in.layerDistanceInMeters,
            pixelGeometry=object_in.getPixelGeometry(),
            center=object_in.getCenter(),
        )
        costs: Sequence[float] = list()

        product = Product(
            metadata=parameters.product.metadata,
            scan=parameters.product.scan,
            probe=parameters.product.probe,
            object_=object_out,
            costs=costs,
        )

        return ReconstructOutput(product, 0)

    def getModelFileFilter(self) -> str:
        return self.MODEL_FILE_FILTER

    def openModel(self, filePath: Path) -> None:
        # FIXME model path to/from settings
        self._inference_settings.model_path.setValue(filePath)
        # ModelManager updates global config (ptycho.params.cfg) when loading
        self._model_dict = ptycho.model_manager.ModelManager.load_multiple_models(
            filePath.parent / filePath.stem
        )
        # FIXME update settings from ptycho.params.cfg after loading

    def saveModel(self, filePath: Path) -> None:
        ptycho.model_manager.save(filePath)

    def getTrainingDataFileFilter(self) -> str:
        return self.TRAINING_DATA_FILE_FILTER

    def exportTrainingData(self, filePath: Path, parameters: ReconstructInput) -> None:
        raw_data = create_raw_data(parameters)
        raw_data.to_file(filePath)

    def getTrainingDataPath(self):
        return self._training_settings.data_dir.getValue()

    def train(self, dataPath: Path) -> TrainOutput:
        self._training_settings.data_dir.setValue(dataPath)

        test_raw_data = RawData.from_file(dataPath / 'test_data.npz')
        train_raw_data = RawData.from_file(dataPath / 'train_data.npz')

        model_size = train_raw_data.diff3d.shape[-1]

        if train_raw_data.diff3d.shape[-2] != model_size:
            raise ValueError('Model requires square diffraction patterns!')

        model_config = self._create_model_config(model_size)
        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path(),  # not used
            test_data_file=None,  # not used
            batch_size=self._training_settings.batch_size.getValue(),
            nepochs=self._training_settings.nepochs.getValue(),
            mae_weight=self._training_settings.mae_weight.getValue(),
            nll_weight=self._training_settings.nll_weight.getValue(),
            realspace_mae_weight=self._training_settings.realspace_mae_weight.getValue(),
            realspace_weight=self._training_settings.realspace_weight.getValue(),
            nphotons=self._training_settings.nphotons.getValue(),  # FIXME
            positions_provided=self._training_settings.positions_provided.getValue(),
            probe_trainable=self._training_settings.probe_trainable.getValue(),
            intensity_scale_trainable=self._training_settings.intensity_scale_trainable.getValue(),
            output_dir=Path(),  # not used
        )

        # Update global params with new-style config
        update_legacy_dict(ptycho.params.cfg, training_config)

        from ptycho.workflows.components import run_cdi_example

        recon_amp, recon_phase, results = run_cdi_example(
            train_raw_data, test_raw_data, training_config
        )  # FIXME verify inputs
        # FIXME update self._model_dict

        return TrainOutput([], [], 0)  # FIXME
