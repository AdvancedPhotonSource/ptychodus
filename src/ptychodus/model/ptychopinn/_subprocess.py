"""Child-side subprocess entry points for the PtychoPINN (TensorFlow) backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that is allowed to import tensorflow or ptycho. The parent-
side ptychodus process never imports this module.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
import zipfile
from collections.abc import Sequence
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any, Final

import numpy

from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstructor import ReconstructOutput, TrainOutput
from ptychodus.api.settings import SettingsRegistry

from ..processing._subprocess_protocol import load_settings_registry_from_string
from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

logger = logging.getLogger(__name__)


MODEL_FILE_NAME: Final[str] = 'wts.h5.zip'


def _rehydrate_settings(
    settings_ini: str,
) -> tuple[
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
    PtychoPINNInferenceSettings,
]:
    registry = SettingsRegistry()
    model_s = PtychoPINNModelSettings(registry)
    training_s = PtychoPINNTrainingSettings(registry)
    inference_s = PtychoPINNInferenceSettings(registry)
    load_settings_registry_from_string(registry, settings_ini)
    return model_s, training_s, inference_s


def _create_model_config(
    model_settings: PtychoPINNModelSettings, name: str, model_size: int
) -> Any:
    from ptycho.config.config import ModelConfig

    return ModelConfig(
        N=model_size,
        gridsize=model_settings.gridsize.get_value(),
        n_filters_scale=model_settings.n_filters_scale.get_value(),
        model_type=name.lower(),
        amp_activation=model_settings.amp_activation.get_value(),
        object_big=model_settings.object_big.get_value(),
        probe_big=model_settings.probe_big.get_value(),
        probe_mask=model_settings.probe_mask.get_value(),
        pad_object=model_settings.pad_object.get_value(),
        probe_scale=model_settings.probe_scale.get_value(),
        gaussian_smoothing_sigma=model_settings.gaussian_smoothing_sigma.get_value(),
    )


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


def _resolve_bundle_dir(model_bundle_path: Path) -> Path:
    """Resolve the parent-recorded model path to a directory the child can load from.

    Accepts either the bundle directory itself, a ``wts.h5.zip`` inside a bundle
    directory, or an outer zip archive to unpack into a fresh tempdir. Mirrors
    the in-process behavior that used to live on the reconstructor.
    """
    if model_bundle_path.name == MODEL_FILE_NAME:
        return model_bundle_path.parent
    if model_bundle_path.suffix == '.zip':
        bundle_dir = Path(tempfile.mkdtemp(prefix='ptychopinn-bundle-'))
        with zipfile.ZipFile(model_bundle_path) as archive:
            archive.extractall(bundle_dir)
        return bundle_dir
    logger.warning(
        f'PtychoPINN expected the file name {MODEL_FILE_NAME!r}; got {model_bundle_path.name!r}.'
    )
    return model_bundle_path.parent


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for inference."""
    if payload.model_bundle_path is None:
        raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

    from ptycho.config.config import InferenceConfig, update_legacy_dict
    from ptycho.workflows.components import load_inference_bundle
    import ptycho.loader
    import ptycho.params
    import ptycho.probe
    import ptycho.tf_helper

    model_s, _, inference_s = _rehydrate_settings(payload.settings_ini)

    parameters = payload.reconstruct_input
    model_size = parameters.diffraction_patterns.shape[-1]

    if parameters.diffraction_patterns.shape[-2] != model_size:
        raise ValueError('Model requires square diffraction patterns!')

    model_config = _create_model_config(model_s, payload.name, model_size)
    inference_config = InferenceConfig(
        model=model_config,
        model_path=Path(),  # not used
        test_data_file=Path(),  # not used
        debug=payload.is_developer_mode_enabled,
        output_dir=Path(),  # not used
    )
    update_legacy_dict(ptycho.params.cfg, inference_config)

    # Load model from the parent-recorded bundle path.
    bundle_dir = _resolve_bundle_dir(payload.model_bundle_path)
    model_obj, _config = load_inference_bundle(bundle_dir)

    test_raw_data = _create_raw_data(parameters)
    ptycho.probe.set_probe_guess(None, test_raw_data.probeGuess)

    test_dataset = test_raw_data.generate_grouped_data(
        model_config.N,
        K=inference_s.n_nearest_neighbors.get_value(),
        nsamples=inference_s.n_samples.get_value(),
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
    from ptycho.config.config import TrainingConfig, update_legacy_dict
    from ptycho.raw_data import RawData
    from ptycho.workflows.components import run_cdi_example, save_outputs
    import ptycho.model_manager
    import ptycho.params

    model_s, training_s, _ = _rehydrate_settings(payload.settings_ini)

    test_raw_data = RawData.from_file(payload.input_path / 'test_data.npz')
    train_raw_data = RawData.from_file(payload.input_path / 'train_data.npz')

    model_size = train_raw_data.diff3d.shape[-1]
    if train_raw_data.diff3d.shape[-2] != model_size:
        raise ValueError('Model requires square diffraction patterns!')

    model_config = _create_model_config(model_s, payload.name, model_size)
    training_config = TrainingConfig(
        model=model_config,
        train_data_file=Path(),
        test_data_file=None,
        batch_size=training_s.batch_size.get_value(),
        nepochs=training_s.nepochs.get_value(),
        mae_weight=training_s.mae_weight.get_value(),
        nll_weight=training_s.nll_weight.get_value(),
        realspace_mae_weight=training_s.realspace_mae_weight.get_value(),
        realspace_weight=training_s.realspace_weight.get_value(),
        nphotons=training_s.nphotons.get_value(),
        positions_provided=training_s.positions_provided.get_value(),
        probe_trainable=training_s.probe_trainable.get_value(),
        intensity_scale_trainable=training_s.intensity_scale_trainable.get_value(),
        output_dir=Path(),
    )
    update_legacy_dict(ptycho.params.cfg, training_config)

    recon_amp, recon_phase, train_results = run_cdi_example(
        train_raw_data, test_raw_data, training_config
    )
    model_path = payload.output_path / MODEL_FILE_NAME
    ptycho.model_manager.save(payload.output_path)
    save_outputs(recon_amp, recon_phase, train_results, str(payload.output_path))

    queue.put((TAG_TRAIN_OUTPUT, pickle.dumps(TrainOutput())))
    queue.put((TAG_MODEL_SAVED, str(model_path)))
