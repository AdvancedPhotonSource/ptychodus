"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN.

Zero tensorflow / ptycho imports. All GPU work runs inside a spawned child;
see :mod:`._subprocess` for the child entry points.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.settings import SettingsRegistry

from ..processing._subprocess_protocol import dump_settings_registry_to_string
from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import ReconstructPayload, TrainPayload
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
    settings_registry: SettingsRegistry,
    *,
    is_developer_mode_enabled: bool,
) -> SubprocessReconstructor:
    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        return ReconstructPayload(
            name=name,
            model_bundle_path=loaded_model_path,
            is_developer_mode_enabled=is_developer_mode_enabled,
            settings_ini=dump_settings_registry_to_string(settings_registry),
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        return TrainPayload(
            name=name,
            is_developer_mode_enabled=is_developer_mode_enabled,
            settings_ini=dump_settings_registry_to_string(settings_registry),
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
