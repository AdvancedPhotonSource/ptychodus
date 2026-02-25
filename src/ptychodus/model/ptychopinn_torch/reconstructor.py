from __future__ import annotations
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
import logging

import numpy

from ptycho_torch.api.base_api import (
    ConfigManager,
    DataConfig,
    DatagenConfig,
    DataloaderFormats,
    InferenceConfig,
    InferenceEngine,
    ModelConfig,
    PtychoDataLoader,
    PtychoModel,
    Trainer,
    TrainingConfig,
)
from ptycho_torch.model import PtychoPINN_Lightning

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
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

__all__ = [
    'PtychoPINNTorchTrainableReconstructor',
]

logger = logging.getLogger(__name__)


class PtychoPINNTorchTrainableReconstructor(TrainableReconstructor):
    def __init__(
        self,
        name: str,
        model_settings: PtychoPINNTorchModelSettings,
        inference_settings: PtychoPINNTorchInferenceSettings,
        training_settings: PtychoPINNTorchTrainingSettings,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._name = name
        self._model_settings = model_settings
        self._inference_settings = inference_settings
        self._training_settings = training_settings
        self._is_developer_mode_enabled = is_developer_mode_enabled

        self._model: PtychoModel | None = None

    def _create_config_manager(self) -> ConfigManager:
        data_config = DataConfig()  # FIXME from settings
        model_config = ModelConfig()  # FIXME from settings
        training_config = TrainingConfig()  # FIXME from settings
        inference_config = InferenceConfig()  # FIXME from settings
        datagen_config = DatagenConfig()  # FIXME from settings

        return ConfigManager(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
        )

    @property
    def name(self) -> str:
        return self._name

    def get_progress_goal(self) -> int:
        return 0

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        if self._model is None:
            raise RuntimeError('Model must be loaded before reconstruction.')

        config_manager = self._create_config_manager()  # FIXME reuse config manager
        ptycho_data_dir = Path()  # FIXME
        tensordict_dataloader = PtychoDataLoader(
            data_dir=ptycho_data_dir, config_manager=config_manager, data_format='tensordict'
        )
        ptycho_inference = InferenceEngine(config_manager=config_manager, ptycho_model=self._model)
        object_out_array = ptycho_inference.predict_and_stitch(tensordict_dataloader)

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.squeeze(object_out_array),
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )
        losses: Sequence[LossValue] = list()

        product = Product(
            metadata=parameters.product.metadata,
            probe_positions=parameters.product.probe_positions,
            probes=parameters.product.probes,
            object_=object_out,
            losses=losses,
        )

        yield ReconstructOutput(product)

    def is_model_loaded(self):
        return self._model is not None

    def get_model_file_filter(self) -> str:
        return 'PyTorch Lightning Checkpoint Files (*.ckpt)'

    def load_model_from_file(self, file_path: Path) -> None:
        json_base_path = Path()  # FIXME
        self._model = PtychoModel._load(
            config_manager=self._create_config_manager(),
            strategy='lightning',
            run_path=json_base_path,  # FIXME
            model_class=PtychoPINN_Lightning,
        )

    def get_training_data_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        # TODO extract & share with ptychopinn
        object_geometry = parameters.product.object_.get_geometry()
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()

        for scan_point in parameters.product.probe_positions:
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            position_x_px.append(object_point.coordinate_x_px)
            position_y_px.append(object_point.coordinate_y_px)

        xcoords = numpy.array(position_x_px)
        ycoords = numpy.array(position_y_px)

        numpy.savez(
            file_path,
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=parameters.diffraction_patterns,
            probeGuess=parameters.product.probes.get_probe_no_opr().get_incoherent_mode(0),
            # assume that all patches are from the same object
            objectGuess=parameters.product.object_.get_layer(0),
            scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
        )

    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        config_manager = self._create_config_manager()  # FIXME reuse config manager?
        timestamp = datetime.now()  # FIXME

        lightning_dataloader = PtychoDataLoader(
            data_dir=input_path,
            config_manager=config_manager,
            data_format=DataloaderFormats('lightning_only_module'),
            output_dir=output_path,
            timestamp=timestamp,  # FIXME
        )

        new_ptycho_model = PtychoModel._new_model(
            model=PtychoPINN_Lightning, config_manager=config_manager
        )
        lightning_trainer = Trainer._from_lightning(
            model=new_ptycho_model,
            dataloader=lightning_dataloader,
            orchestration='lightning',
            config_manager=config_manager,
        )
        output_dir = lightning_trainer.train(orchestration='lightning', experiment_name='test_run')

        new_destination = output_path / 'new_ptycho_model'  # FIXME file suffix
        new_ptycho_model.save(
            path=new_destination, source_run_path=output_dir, strategy='lightning'
        )
        self._model = new_ptycho_model

        yield TrainOutput()  # TODO yield losses & progress
