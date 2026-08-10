"""Child-side subprocess entry points for the PtychoPINN-Torch backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that is allowed to import lightning or the GPU-side
``ptycho_torch`` packages (``api.base_api``, ``model``, ``lightning_utils``).
The parent-side ptychodus process never imports this module -- it reaches only
``ptycho_torch.config_params``, from inside :func:`.reconstructor._build_configs`.

Two entry points are exposed:

- :func:`run_reconstruct` -- load a checkpoint, run one inference pass,
  stream back a single :class:`ReconstructOutput`.
- :func:`run_train` -- run one Lightning training session (which may itself
  fan out to ``n_devices`` DDP ranks via ``strategy='ddp_spawn'``), save the
  best checkpoint, and stream back the final :class:`TrainOutput` plus the
  saved-checkpoint path.

Neither entry point reads ptychodus settings. Training receives finished
``ptycho_torch`` config objects on the payload and only assembles them into a
``ConfigManager``; inference rebuilds its configs from the checkpoint.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Sequence
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.diffraction import zero_bad_pixels
from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstructor import ReconstructOutput, TrainOutput

from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import ReconstructPayload, TrainPayload

logger = logging.getLogger(__name__)


def _load_ptycho_model(model_path: Path) -> tuple[Any, Any]:
    """Load a PtychoModel from a .ckpt file. Returns (model, config_manager)."""
    from ptycho_torch.api.base_api import ConfigManager, PtychoModel
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config, model_config, training_config, inference_config = (
        PtychoModel._extract_configs_from_checkpoint(str(model_path))
    )
    if any(c is None for c in (data_config, model_config, training_config, inference_config)):
        raise ValueError(
            f'Checkpoint at {model_path} is missing one or more saved configs '
            f'(data/model/training/inference).'
        )

    ptycho_model = PtychoModel(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    ptycho_model.model = PtychoPINN_Lightning.load_from_checkpoint(
        model_path,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )

    config_manager = ConfigManager.from_loaded_model(ptycho_model)
    config_manager.validate_arch_compatibility(ptycho_model)
    return ptycho_model, config_manager


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for one inference pass. Streams a single ReconstructOutput."""
    if payload.model_path is None:
        raise RuntimeError('Cannot reconstruct: no model checkpoint has been loaded.')

    from ptycho_torch.api.base_api import InferenceEngine, PtychoDataLoader

    ptycho_model, config_manager = _load_ptycho_model(payload.model_path)

    if config_manager.model_config.mode != payload.model_training_mode:
        logger.warning(
            'Loaded checkpoint mode %r does not match reconstructor mode %r; '
            'predictions may be inconsistent.',
            config_manager.model_config.mode,
            payload.model_training_mode,
        )

    inference_engine = InferenceEngine(config_manager=config_manager, ptycho_model=ptycho_model)

    parameters = payload.reconstruct_input
    object_geometry = parameters.product.object_.get_geometry()
    positions_px: list[float] = list()

    for position in parameters.product.probe_positions:
        object_point = object_geometry.map_coordinates_probe_to_object(position)
        positions_px.append(object_point.coordinate_y_px)
        positions_px.append(object_point.coordinate_x_px)

    diff_patterns = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
    data_loader = PtychoDataLoader.from_np(
        diff_patterns=diff_patterns,
        probe=parameters.product.probes.get_probe_no_opr().get_array(),
        positions=numpy.reshape(positions_px, (-1, 2)),
        config_manager=config_manager,
    )
    object_out_array = numpy.asarray(inference_engine.predict_and_stitch(data_loader))

    object_in = parameters.product.object_
    object_out = Object(
        array=object_out_array,
        layer_spacing_m=object_in.layer_spacing_m,
        pixel_geometry=object_in.get_pixel_geometry(),
        center=object_in.get_center(),
    )

    losses: Sequence[LossValue] = []
    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_out,
        losses=losses,
    )

    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product=product, progress=1))))


def _apply_visible_devices(visible_gpu_indices: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before the first CUDA call in this process.

    Torch has already been imported by the time this runs -- unpickling the
    payload pulls in ``ptycho_torch.config_params``, which imports it. That is
    fine: importing torch reads no device list. The driver resolves
    CUDA_VISIBLE_DEVICES when the runtime is first touched, so masking works as
    long as nothing has called ``torch.cuda.*`` yet. Keep this the first
    statement of :func:`run_train`.
    """
    trimmed = visible_gpu_indices.strip()
    if trimmed:
        os.environ['CUDA_VISIBLE_DEVICES'] = trimmed


def _clamp_n_devices(requested: int) -> int:
    """Return min(requested, torch.cuda.device_count()); warn on clamp."""
    import torch

    available = torch.cuda.device_count()
    if available == 0:
        logger.warning('No CUDA devices available; training will fall back to CPU.')
        return 1
    if requested > available:
        logger.warning(
            'Requested n_devices=%d but only %d CUDA device(s) available; clamping.',
            requested,
            available,
        )
        return available
    return requested


def run_train(payload: TrainPayload, queue: Queue[Any]) -> None:
    """Child entry point for one training session.

    Masks the visible GPUs, clamps ``n_devices`` to what's actually visible,
    then runs the Lightning training loop with the configured DDP strategy.
    The configs themselves were built parent-side; all this does with them is
    assemble the ``ConfigManager``.
    """
    _apply_visible_devices(payload.visible_gpu_indices)

    from lightning.pytorch.callbacks import Callback
    from ptycho_torch.api.base_api import (
        ConfigManager,
        DataloaderFormats,
        PtychoDataLoader,
        PtychoModel,
        Trainer,
    )
    from ptycho_torch.lightning_utils import find_best_checkpoint
    from ptycho_torch.model import PtychoPINN_Lightning

    training_config = payload.training_config
    training_config.n_devices = _clamp_n_devices(training_config.n_devices)

    config_manager = ConfigManager(
        data_config=payload.data_config,
        model_config=payload.model_config,
        training_config=training_config,
        inference_config=payload.inference_config,
        datagen_config=payload.datagen_config,
    )

    class _LossCollectorCallback(Callback):
        """Collects per-epoch train and validation losses from the Lightning Trainer."""

        def __init__(self, train_metric_name: str, val_metric_name: str) -> None:
            super().__init__()
            self._train_metric_name = train_metric_name
            self._val_metric_name = val_metric_name
            self.training_loss: list[LossValue] = []
            self.validation_loss: list[LossValue] = []
            self.epochs_completed = 0

        def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            value = trainer.callback_metrics.get(self._train_metric_name)
            if value is not None:
                self.training_loss.append(
                    LossValue(epoch=trainer.current_epoch, value=float(value))
                )
            self.epochs_completed = trainer.current_epoch + 1

        def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            if trainer.sanity_checking:
                return
            value = trainer.callback_metrics.get(self._val_metric_name)
            if value is not None:
                self.validation_loss.append(
                    LossValue(epoch=trainer.current_epoch, value=float(value))
                )

    data_loader = PtychoDataLoader(
        data_dir=payload.input_path,
        config_manager=config_manager,
        data_format=DataloaderFormats('lightning_only_module'),
        output_dir=payload.output_path,
    )
    model = PtychoModel._new_model(model=PtychoPINN_Lightning, config_manager=config_manager)
    trainer = Trainer._from_lightning(
        model=model,
        dataloader=data_loader,
        orchestration='lightning',
        config_manager=config_manager,
    )

    loss_collector = _LossCollectorCallback(
        train_metric_name=model.model.loss_name,
        val_metric_name=model.model.val_loss_name,
    )
    trainer._trainer.callbacks.append(loss_collector)

    trainer.train(
        orchestration='lightning',
        experiment_name='',
    )
    # PtychoDataLoader appends a `run_<timestamp>` segment to output_dir,
    # and the checkpoint callback writes there — not at output_path.
    run_dir = Path(data_loader.output_dir)
    checkpoint_path = find_best_checkpoint(run_dir)

    if checkpoint_path is None:
        raise FileNotFoundError(f'No checkpoints found in {run_dir} after training.')

    queue.put(
        (
            TAG_TRAIN_OUTPUT,
            pickle.dumps(
                TrainOutput(
                    training_loss=loss_collector.training_loss,
                    validation_loss=loss_collector.validation_loss,
                    progress=loss_collector.epochs_completed,
                )
            ),
        )
    )
    queue.put((TAG_MODEL_SAVED, str(checkpoint_path)))
