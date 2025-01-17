from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any
import logging

import numpy
import numpy.typing

import ptycho.config
import ptycho.loader
import ptycho.model_manager
import ptycho.params
import ptycho.raw_data

from ptychodus.api.object import Object, ObjectArrayType
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


def create_raw_data(
    parameters: ReconstructInput, *, use_object_guess: bool
) -> ptycho.raw_data.RawData:
    object_geometry = parameters.product.object_.getGeometry()
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in parameters.product.scan:
        object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
        position_x_px.append(object_point.positionXInPixels)
        position_y_px.append(object_point.positionYInPixels)

    objectGuess: ObjectArrayType | None = None

    if use_object_guess:
        objectGuess = parameters.product.object_.getLayer(0)

    return ptycho.raw_data.RawData.from_coords_without_pc(
        xcoords=numpy.array(position_x_px),
        ycoords=numpy.array(position_y_px),
        diff3d=parameters.patterns,
        probeGuess=parameters.product.probe.getIncoherentMode(0),
        # Assuming all patches are from the same object
        scan_index=numpy.zeros(len(parameters.product.scan)),  # FIXME
        objectGuess=objectGuess,
    )


class PtychoPINNTrainableReconstructor(TrainableReconstructor):
    def __init__(
        self,
        name: str,
        model_settings: PtychoPINNModelSettings,
        training_settings: PtychoPINNTrainingSettings,
        inference_settings: PtychoPINNInferenceSettings,
    ) -> None:
        super().__init__()
        self._name = name
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._inference_settings = inference_settings

        self._training_data_file_filters = ['NumPy Zipped Archive (*.npz)']
        self._model_file_filters = ['Zipped Archive (*.zip)']
        self._model_dict: dict[str, Any] | None = None

        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        solution_region_size = self._model_config['N']  # FIXME ptycho.params.cfg or settings
        num_nearest_neighbors = 7
        num_samples = 1

        # xpp dataset
        # "u" dataset
        # ALS dataset
        # normalize data in preprocessing step (see note in slack)
        # coords in pixel units
        # origin is CoM coords of solution regions; origin doesn't matter
        # ptychodus stitches
        # supervised parameters are subset of PINN
        #   - hide all except #filt
        #   - all losses out; positions provided/trainable, intensity
        #   - just batch size and n# epocs in training
        # tool tips not working

        # ModelConfig -> InferenceConfig
        inferenceConfig = ptycho.config.config.InferenceConfig(modelConfig)  # FIXME where used?

        # Update global params with new-style config
        ptycho.config.config.update_legacy_dict(
            ptycho.params.cfg, inferenceConfig
        )  # FIXME investigate

        ptycho.probe.set_probe_guess(None, test_data.probeGuess)  # FIXME ???

        # Create RawData
        test_data = create_raw_data(parameters, use_object_guess=False)
        test_dataset = test_data.generate_grouped_data(
            solution_region_size, K=num_nearest_neighbors, nsamples=num_samples
        )

        # Create PtychoDataContainer
        test_data_container = ptycho.loader.load(
            lambda: test_dataset, test_data.probeGuess, which=None, create_split=False
        )

        # Perform reconstruction
        model = self._model_dict['diffraction_to_obj']  # tf.keras.Model
        obj_tensor_full = model.predict(
            [test_data.X * model.params()['intensity_scale'], test_data.local_offsets]
        )

        # Process the reconstructed image
        object_out_array = ptycho.tf_helper.reassemble_position(
            obj_tensor_full, test_data.global_offsets, M=20
        )

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.array(object_out_array),
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

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass  # FIXME

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._training_data_file_filters

    def getOpenTrainingDataFileFilter(self) -> str:
        return self._training_data_file_filters[0]

    def openTrainingData(self, filePath: Path) -> None:
        pass  # FIXME

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._training_data_file_filters

    def getSaveTrainingDataFileFilter(self) -> str:
        return self._training_data_file_filters[0]

    def saveTrainingData(self, filePath: Path) -> None:
        pass  # FIXME

    def train(self) -> TrainOutput:
        from ptycho.workflows.components import (
            setup_configuration,
            load_data,
            run_cdi_example,
            save_outputs,
        )
        from ptycho.config.config import TrainingConfig, update_legacy_dict

        config = setup_configuration(args, args.config)

        # Update global params with new-style config at entry point
        ptycho.config.config.update_legacy_dict(ptycho.params.cfg, config)

        # ptycho_data, ptycho_data_train, obj = load_and_prepare_data(config['train_data_file_path'])
        ptycho_data = load_data(str(config.train_data_file), n_images=512)

        test_data = None

        if config.test_data_file:
            test_data = load_data(str(config.test_data_file))

        recon_amp, recon_phase, results = run_cdi_example(ptycho_data, test_data, config)
        ptycho.model_manager.save(str(config.output_dir))
        save_outputs(recon_amp, recon_phase, results, str(config.output_dir))

        return TrainOutput([], [], 0)  # FIXME

    def clearTrainingData(self) -> None:
        pass  # FIXME

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        return self._model_file_filters

    def getOpenModelFileFilter(self) -> str:
        return self._model_file_filters[0]

    def openModel(self, filePath: Path) -> None:
        # ModelManager updates global config (ptycho.params.cfg) when loading
        self._model_dict = ptycho.model_manager.ModelManager.load_multiple_models(filePath)
        # FIXME model path to/from settings

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return self._model_file_filters

    def getSaveModelFileFilter(self) -> str:
        return self._model_file_filters[0]

    def saveModel(self, filePath: Path) -> None:
        ptycho.model_manager.save(filePath)
