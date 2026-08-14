"""Child-side subprocess entry points for the PtychoNN backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that is allowed to import ptychonn / torch / lightning.

The child receives a small pydantic config on the payload and reads the
scalars it needs directly — it does not touch ``SettingsRegistry`` or the
parent's settings-class layout.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from multiprocessing.queues import Queue
from typing import Any

import numpy

from ptychodus.api.interpolate import BarycentricArrayStitcher
from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import ReconstructOutput, TrainOutput

from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import (
    PtychoNNReconstructConfig,
    ReconstructPayload,
    TrainPayload,
)

logger = logging.getLogger(__name__)


PATCHES_KEY = 'real'
PATTERNS_KEY = 'reciprocal'


def _build_model(
    config: PtychoNNReconstructConfig,
    *,
    checkpoint_path: Any = None,
) -> Any:
    import ptychonn

    if checkpoint_path is not None:
        return ptychonn.LitReconSmallModel.load_from_checkpoint(checkpoint_path)
    return ptychonn.LitReconSmallModel(
        nconv=config.num_convolution_kernels,
        use_batch_norm=config.use_batch_normalization,
        enable_amplitude=config.enable_amplitude,
        max_lr=config.max_learning_rate,
        min_lr=config.min_learning_rate,
    )


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for one PtychoNN inference pass."""
    import ptychonn

    model = _build_model(payload.config, checkpoint_path=payload.model_path)

    parameters = payload.reconstruct_input
    data = parameters.diffraction_patterns
    data_size = data.shape[-1]
    if data_size != data.shape[-2]:
        raise ValueError('PtychoNN expects square diffraction data!')
    is_data_size_pow2 = data_size & (data_size - 1) == 0 and data_size > 0
    if not is_data_size_pow2:
        raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

    logger.debug('Inferring...')
    object_patches = ptychonn.infer(data=data.astype(numpy.float32), model=model)

    logger.debug('Stitching...')
    object_array = parameters.product.object_.get_array()
    object_geometry = parameters.product.object_.get_geometry()
    stitcher = BarycentricArrayStitcher(
        upper=numpy.zeros_like(object_array),
        lower=numpy.zeros_like(object_array, dtype=float),
    )
    for scan_point, object_patch_channels in zip(
        parameters.product.probe_positions, object_patches
    ):
        patch_array = numpy.exp(1j * object_patch_channels[0])
        if object_patch_channels.shape[0] == 2:
            patch_array *= object_patch_channels[1]
        else:
            patch_array *= 0.5
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        stitcher.add_patch(object_point.coordinate_x_px, object_point.coordinate_y_px, patch_array)

    object_ = Object(
        array=stitcher.stitch(),
        pixel_geometry=object_geometry.get_pixel_geometry(),
        center=object_geometry.get_center(),
        layer_spacing_m=parameters.product.object_.layer_spacing_m,
    )
    losses: Sequence[LossValue] = list()

    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_,
        losses=losses,
    )
    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product))))


def run_train(payload: TrainPayload, queue: Queue[Any]) -> None:
    """Child entry point for one PtychoNN training session."""
    import ptychonn

    config = payload.config

    logger.debug(f'Reading "{payload.input_path}" as "NPZ"')
    training_data = numpy.load(payload.input_path)

    model = _build_model(config)
    training_set_fractional_size = 1 - config.validation_set_fractional_size
    trainer, trainer_log = ptychonn.train(
        model=model,
        batch_size=config.batch_size,
        out_dir=None,
        X_train=training_data[PATTERNS_KEY],
        Y_train=training_data[PATCHES_KEY],
        epochs=config.training_epochs,
        training_fraction=training_set_fractional_size,
        log_frequency=config.status_interval_in_epochs,
        strategy='ddp_notebook',
    )

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
            validation_loss.append(vloss)

    checkpoint_path = payload.output_path
    trainer.save_checkpoint(checkpoint_path)

    queue.put(
        (
            TAG_TRAIN_OUTPUT,
            pickle.dumps(TrainOutput(training_loss=training_loss, validation_loss=validation_loss)),
        )
    )
    queue.put((TAG_MODEL_SAVED, str(checkpoint_path)))
