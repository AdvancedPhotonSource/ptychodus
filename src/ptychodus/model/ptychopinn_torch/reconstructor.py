"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN-Torch.

This module has ZERO torch / lightning / ptycho_torch imports. All GPU work
runs inside a spawned child; see :mod:`._subprocess` for the child entry
points.
"""

from __future__ import annotations

from pathlib import Path

from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.settings import SettingsRegistry

from ..processing._subprocess_protocol import dump_settings_registry_to_string
from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

__all__ = [
    'build_reconstructor',
]


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychopinn_torch._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptychopinn_torch._subprocess:run_train'


def build_reconstructor(
    model_training_mode: str,
    data_settings: PtychoPINNTorchDataSettings,
    model_settings: PtychoPINNTorchModelSettings,
    inference_settings: PtychoPINNTorchInferenceSettings,
    training_settings: PtychoPINNTorchTrainingSettings,
    settings_registry: SettingsRegistry,
    *,
    is_developer_mode_enabled: bool,
) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoPINN-Torch mode.

    ``model_training_mode`` is 'Unsupervised' or 'Supervised' and becomes the
    reconstructor's display name plus the config-manager mode string used
    inside the child.
    """

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        return ReconstructPayload(
            model_training_mode=model_training_mode,
            model_path=loaded_model_path,
            is_developer_mode_enabled=is_developer_mode_enabled,
            settings_ini=dump_settings_registry_to_string(settings_registry),
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        return TrainPayload(
            model_training_mode=model_training_mode,
            is_developer_mode_enabled=is_developer_mode_enabled,
            settings_ini=dump_settings_registry_to_string(settings_registry),
            input_path=input_path,
            output_path=output_path,
            n_devices=training_settings.n_devices.get_value(),
            distributed_strategy=training_settings.distributed_strategy.get_value(),
            visible_gpu_indices=training_settings.visible_gpu_indices.get_value(),
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        save_ptychopinn_training_data(file_path, parameters, multimodal_probe=True)

    return SubprocessReconstructor(
        name=model_training_mode,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        progress_goal_fn=lambda: training_settings.epochs.get_value(),
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='PyTorch Lightning Checkpoint Files (*.ckpt)',
        model_file_extension='.ckpt',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
    )
