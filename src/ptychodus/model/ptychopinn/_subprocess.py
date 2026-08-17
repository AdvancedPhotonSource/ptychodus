"""Child-side subprocess entry points for the PtychoPINN (TensorFlow) backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that is allowed to import tensorflow or the TensorFlow-backed
``ptycho`` submodules. The parent-side ptychodus process never imports this
module.

The payload carries finished ``ptycho`` config objects and an already-unpacked
bundle directory (see :mod:`._payload`), so the child reads no ptychodus
settings and does no archive handling -- it converts the input product to
``RawData``, runs the model, and converts the result back.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from multiprocessing.queues import Queue
from typing import Any

import numpy

from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import ReconstructOutput, TrainOutput

from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import MODEL_FILE_NAME, ReconstructPayload, TrainPayload

logger = logging.getLogger(__name__)


def _create_raw_data(parameters: Any) -> Any:
    from ptycho.raw_data import RawData

    object_geometry = parameters.product.object_.get_geometry()
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in parameters.product.probe_positions:
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        position_x_px.append(object_point.coordinate_x_px)
        position_y_px.append(object_point.coordinate_y_px)

    return RawData.from_coords_without_pc(
        xcoords=numpy.array(position_x_px),
        ycoords=numpy.array(position_y_px),
        diff3d=parameters.diffraction_patterns,
        probeGuess=parameters.product.probes.get_probe_no_opr().get_incoherent_mode(0),
        scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
        objectGuess=parameters.product.object_.get_layer(0),
    )


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for inference."""
    from ptycho.config.config import update_legacy_dict
    from ptycho.workflows.components import load_inference_bundle
    import ptycho.loader
    import ptycho.params
    import ptycho.probe
    import ptycho.tf_helper

    inference_config = payload.inference_config
    update_legacy_dict(ptycho.params.cfg, inference_config)

    model_obj, _config = load_inference_bundle(payload.model_bundle_dir)

    parameters = payload.reconstruct_input
    test_raw_data = _create_raw_data(parameters)
    ptycho.probe.set_probe_guess(None, test_raw_data.probeGuess)

    test_dataset = test_raw_data.generate_grouped_data(
        inference_config.model.N,
        K=payload.n_nearest_neighbors,
        nsamples=payload.n_samples,
    )
    test_data_container = ptycho.loader.load(
        lambda: test_dataset, test_raw_data.probeGuess, which=None, create_split=False
    )

    try:
        intensity_scale = ptycho.params.get('intensity_scale')
    except KeyError as exc:
        raise RuntimeError('Missing intensity_scale in ptycho.params.cfg') from exc

    obj_tensor_full = model_obj.predict(
        [test_data_container.X * intensity_scale, test_data_container.local_offsets]
    )
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
    losses: Sequence[LossValue] = list()
    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_out,
        losses=losses,
    )

    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product))))


def run_train(payload: TrainPayload, queue: Queue[Any]) -> None:
    """Child entry point for training."""
    from ptycho.config.config import update_legacy_dict
    from ptycho.raw_data import RawData
    from ptycho.workflows.components import run_cdi_example, save_outputs
    import ptycho.model_manager
    import ptycho.params

    test_raw_data = RawData.from_file(payload.input_path / 'test_data.npz')
    train_raw_data = RawData.from_file(payload.input_path / 'train_data.npz')

    model_size = train_raw_data.diff3d.shape[-1]
    if train_raw_data.diff3d.shape[-2] != model_size:
        raise ValueError('Model requires square diffraction patterns!')

    # Only the child sees the training arrays, so it is the only place that can
    # resolve the model size the parent left unset.
    training_config = payload.training_config
    training_config.model.N = model_size
    update_legacy_dict(ptycho.params.cfg, training_config)

    recon_amp, recon_phase, train_results = run_cdi_example(
        train_raw_data, test_raw_data, training_config
    )
    model_path = payload.output_path / MODEL_FILE_NAME
    ptycho.model_manager.save(payload.output_path)
    save_outputs(recon_amp, recon_phase, train_results, str(payload.output_path))

    queue.put((TAG_TRAIN_OUTPUT, pickle.dumps(TrainOutput())))
    queue.put((TAG_MODEL_SAVED, str(model_path)))
