from importlib.metadata import version
from pathlib import Path
from typing import Final
import logging

import numpy
import ptychonn

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    LossValue,
    ReconstructInput,
    ReconstructOutput,
    TrainOutput,
    TrainableReconstructor,
)
from ptychodus.api.typing import ComplexArrayType

from ..analysis import BarycentricArrayInterpolator, BarycentricArrayStitcher
from .model import PtychoNNModelProvider
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class CenterBoxMeanPhaseCenteringStrategy:  # TODO USE
    def __call__(self, array: ComplexArrayType) -> ComplexArrayType:
        one_third_height = array.shape[-2] // 3
        one_third_width = array.shape[-1] // 3

        amplitude = numpy.absolute(array)
        phase = numpy.angle(array)

        center_box_mean_phase = phase[
            one_third_height : one_third_height * 2, one_third_width : one_third_width * 2
        ].mean()

        return amplitude * numpy.exp(1j * (phase - center_box_mean_phase))


class PtychoNNTrainableReconstructor(TrainableReconstructor):
    MODEL_FILE_FILTER: Final[str] = 'PyTorch Lightning Checkpoint Files (*.ckpt)'
    TRAINING_DATA_FILE_FILTER: Final[str] = 'NumPy Zipped Archive (*.npz)'
    PATCHES_KEY: Final[str] = 'real'
    PATTERNS_KEY: Final[str] = 'reciprocal'

    def __init__(
        self,
        model_settings: PtychoNNModelSettings,
        training_settings: PtychoNNTrainingSettings,
        model_provider: PtychoNNModelProvider,
    ) -> None:
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._model_provider = model_provider

        ptychonn_version = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonn_version}')

    @property
    def name(self) -> str:
        return self._model_provider.get_model_name()

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        # TODO data size/shape requirements to GUI
        data = parameters.diffraction_patterns
        data_size = data.shape[-1]

        if data_size != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        is_data_size_pow2 = data_size & (data_size - 1) == 0 and data_size > 0

        if not is_data_size_pow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        model = self._model_provider.get_model()

        logger.debug('Inferring...')
        object_patches = ptychonn.infer(
            data=data.astype(numpy.float32),
            model=model,
        )

        logger.debug('Stitching...')
        object_array = parameters.product.object_.get_array()
        object_geometry = parameters.product.object_.get_geometry()
        stitcher = BarycentricArrayStitcher(
            upper=numpy.zeros_like(object_array), lower=numpy.zeros_like(object_array, dtype=float)
        )

        for scan_point, object_patch_channels in zip(parameters.product.positions, object_patches):
            patch_array = numpy.exp(1j * object_patch_channels[0])

            if object_patch_channels.shape[0] == 2:
                patch_array *= object_patch_channels[1]
            else:
                patch_array *= 0.5

            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            stitcher.add_patch(object_point.position_x_px, object_point.position_y_px, patch_array)

        object_ = Object(
            array=stitcher.stitch(),
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
            layer_spacing_m=parameters.product.object_.layer_spacing_m,
        )

        product = Product(
            metadata=parameters.product.metadata,
            positions=parameters.product.positions,
            probes=parameters.product.probes,
            object_=object_,
            losses=list(),  # TODO put something here?
        )

        return ReconstructOutput(product, 0)

    def get_model_file_filter(self) -> str:
        return self.MODEL_FILE_FILTER

    def open_model(self, file_path: Path) -> None:
        self._model_provider.open_model(file_path)

    def save_model(self, file_path: Path) -> None:
        self._model_provider.save_model(file_path)

    def get_training_data_file_filter(self) -> str:
        return self.TRAINING_DATA_FILE_FILTER

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        object_geometry = parameters.product.object_.get_geometry()
        interpolator = BarycentricArrayInterpolator(parameters.product.object_.get_array())
        num_channels = self._model_provider.get_num_channels()
        probe_extent = ImageExtent(
            width_px=parameters.product.probes.width_px,
            height_px=parameters.product.probes.height_px,
        )
        patches = numpy.zeros(
            (len(parameters.product.positions), num_channels, *probe_extent.shape),
            dtype=numpy.float32,
        )

        for index, scan_point in enumerate(parameters.product.positions):
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            patch = interpolator.get_patch(
                object_point.position_x_px,
                object_point.position_y_px,
                probe_extent.width_px,
                probe_extent.height_px,
            )
            patches[index, 0, :, :] = numpy.angle(patch)

            if num_channels > 1:
                patches[index, 1, :, :] = numpy.absolute(patch)

        logger.debug(f'Writing "{file_path}" as "NPZ"')
        contents = {
            self.PATTERNS_KEY: parameters.diffraction_patterns.astype(numpy.float32),
            self.PATCHES_KEY: patches,
        }
        numpy.savez_compressed(file_path, allow_pickle=False, **contents)

    def get_training_data_path(self) -> Path:
        return self._training_settings.training_data_path.get_value()

    def train(self, data_path: Path) -> TrainOutput:
        logger.debug(f'Reading "{data_path}" as "NPZ"')
        training_data = numpy.load(data_path)
        self._training_settings.training_data_path.set_value(data_path)

        model = self._model_provider.get_model()
        logger.debug('Training...')
        training_set_fractional_size = (
            1 - self._training_settings.validation_set_fractional_size.get_value()
        )
        trainer, trainer_log = ptychonn.train(
            model=model,
            batch_size=self._model_settings.batch_size.get_value(),
            out_dir=None,
            X_train=training_data[self.PATTERNS_KEY],
            Y_train=training_data[self.PATCHES_KEY],
            epochs=self._training_settings.training_epochs.get_value(),
            training_fraction=float(training_set_fractional_size),
            log_frequency=self._training_settings.status_interval_in_epochs.get_value(),
            strategy='ddp_notebook',
        )
        self._model_provider.set_trainer(trainer)

        training_loss: list[LossValue] = []
        validation_loss: list[LossValue] = []

        for epoch, entry in enumerate(trainer_log.logs):
            try:
                tloss = LossValue(epoch, entry['training_loss'])
                vloss = LossValue(epoch, entry['validation_loss'])
            except KeyError:
                pass
            else:
                training_loss.append(tloss)
                training_loss.append(vloss)

        return TrainOutput(
            training_loss=training_loss,
            validation_loss=validation_loss,
            result=0,
        )
