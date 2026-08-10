"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN.

Zero tensorflow imports. All GPU work runs inside a spawned child; see
:mod:`._subprocess` for the child entry points.

This module does reach ``ptycho.config.config`` -- but only from inside the
config builders, so the import happens on the first reconstruct/train call
rather than at composition-root time. That subpackage is TensorFlow-free; see
the note in :mod:`._payload` for why it is allowed parent-side.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.reconstructor import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import MODEL_FILE_NAME, ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

__all__ = [
    'build_reconstructor',
]

logger = logging.getLogger(__name__)


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychopinn._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptychopinn._subprocess:run_train'


def _build_model_config(model_settings: PtychoPINNModelSettings, name: str, model_size: int) -> Any:
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


def _build_inference_config(
    model_settings: PtychoPINNModelSettings,
    name: str,
    model_size: int,
    *,
    is_developer_mode_enabled: bool,
) -> Any:
    from ptycho.config.config import InferenceConfig

    return InferenceConfig(
        model=_build_model_config(model_settings, name, model_size),
        model_path=Path(),  # not used
        test_data_file=Path(),  # not used
        debug=is_developer_mode_enabled,
        output_dir=Path(),  # not used
    )


def _build_training_config(
    model_settings: PtychoPINNModelSettings,
    training_settings: PtychoPINNTrainingSettings,
    name: str,
) -> Any:
    from ptycho.config.config import TrainingConfig

    # ``N`` is unknown here: it comes from the training data the child loads.
    # The child overwrites ``model.N`` before using this config.
    return TrainingConfig(
        model=_build_model_config(model_settings, name, model_size=0),
        train_data_file=Path(),
        test_data_file=None,
        batch_size=training_settings.batch_size.get_value(),
        nepochs=training_settings.nepochs.get_value(),
        mae_weight=training_settings.mae_weight.get_value(),
        nll_weight=training_settings.nll_weight.get_value(),
        realspace_mae_weight=training_settings.realspace_mae_weight.get_value(),
        realspace_weight=training_settings.realspace_weight.get_value(),
        nphotons=training_settings.nphotons.get_value(),
        positions_provided=training_settings.positions_provided.get_value(),
        probe_trainable=training_settings.probe_trainable.get_value(),
        intensity_scale_trainable=training_settings.intensity_scale_trainable.get_value(),
        output_dir=Path(),
    )


def _extract_bundle_dir(model_bundle_path: Path) -> Path:
    """Resolve the recorded model path to a directory the child can load from.

    Accepts the bundle directory itself, a ``wts.h5.zip`` inside a bundle
    directory, or an outer zip archive to unpack into a fresh tempdir. Runs
    parent-side because it is pure zipfile/tempfile work with no GPU
    involvement.
    """
    if model_bundle_path.name == MODEL_FILE_NAME:
        return model_bundle_path.parent

    if model_bundle_path.suffix == '.zip':
        bundle_dir = Path(tempfile.mkdtemp(prefix='ptychopinn-bundle-'))
        logger.debug(f'Extracting bundle "{model_bundle_path}" -> "{bundle_dir}"')

        with zipfile.ZipFile(model_bundle_path) as archive:
            archive.extractall(bundle_dir)

        return bundle_dir

    logger.warning(
        f'PtychoPINN expected the file name {MODEL_FILE_NAME!r}; got {model_bundle_path.name!r}.'
    )
    return model_bundle_path.parent


def _save_model_bundle(loaded_from: Path, dest: Path) -> None:
    """Archive the loaded bundle directory (or copy the bundle .zip) to ``dest``.

    The child records either a bundle directory (right after training) or a
    ``wts.h5.zip`` file (right after inference load). This function normalises
    to a zip archive at ``dest``.
    """
    if loaded_from.is_dir():
        archive_stem = str(dest.with_suffix(''))
        logger.debug(f'Archiving bundle {loaded_from!r} -> {dest!r}')
        shutil.make_archive(archive_stem, 'zip', root_dir=loaded_from)
        return

    if loaded_from.suffix == '.zip':
        shutil.copyfile(loaded_from, dest)
        return

    # wts.h5.zip inside a bundle dir — archive the parent dir.
    if loaded_from.name.endswith('.zip'):
        parent = loaded_from.parent
        archive_stem = str(dest.with_suffix(''))
        shutil.make_archive(archive_stem, 'zip', root_dir=parent)
        return

    raise RuntimeError(f'Cannot save PtychoPINN model: unrecognized source path {loaded_from!r}.')


def build_reconstructor(
    name: str,
    model_settings: PtychoPINNModelSettings,
    inference_settings: PtychoPINNInferenceSettings,
    training_settings: PtychoPINNTrainingSettings,
    *,
    is_developer_mode_enabled: bool,
) -> SubprocessReconstructor:
    # Source model path -> extracted bundle directory, so repeated reconstructs
    # against the same model unpack the archive only once.
    bundle_dir_cache: dict[Path, Path] = dict()

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        if loaded_model_path is None:
            raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

        model_size = parameters.diffraction_patterns.shape[-1]

        if parameters.diffraction_patterns.shape[-2] != model_size:
            raise ValueError('Model requires square diffraction patterns!')

        try:
            bundle_dir = bundle_dir_cache[loaded_model_path]
        except KeyError:
            bundle_dir = _extract_bundle_dir(loaded_model_path)
            bundle_dir_cache[loaded_model_path] = bundle_dir

        return ReconstructPayload(
            inference_config=_build_inference_config(
                model_settings,
                name,
                model_size,
                is_developer_mode_enabled=is_developer_mode_enabled,
            ),
            model_bundle_dir=bundle_dir,
            n_nearest_neighbors=inference_settings.n_nearest_neighbors.get_value(),
            n_samples=inference_settings.n_samples.get_value(),
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        return TrainPayload(
            training_config=_build_training_config(model_settings, training_settings, name),
            input_path=input_path,
            output_path=output_path,
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        save_ptychopinn_training_data(file_path, parameters, multimodal_probe=False)

    return SubprocessReconstructor(
        name=name,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='Zipped Archive (*.zip)',
        model_file_extension='.zip',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
        save_model=_save_model_bundle,
    )
