"""Payload dataclasses for the PtychoPINN-Torch subprocess entry points.

``ptycho_torch`` is an optional extra, so it is imported under
:data:`~typing.TYPE_CHECKING` only: this module has ``from __future__ import
annotations`` and both payloads are dataclasses, whose field annotations are
never evaluated. That keeps ``ptychodus.model`` importable without the extra
installed. The factory defers the matching runtime import into
``_build_configs``.

Unlike ``ptycho.config.config``, ``ptycho_torch.config_params`` does pull torch
into the parent (``ptycho_torch/__init__.py`` imports it unconditionally). That
is permitted -- importing torch acquires no GPU context -- and it is the same
bargain ptychi makes for ``PtychographyTaskOptions``. The cost lands on the
first training call rather than at startup because the import lives inside the
payload builder. Inference needs no config at all, so it never pays it.

Everything here is picklable: the configs are plain dataclasses of scalars, and
:class:`ReconstructInput` is a frozen dataclass of numpy arrays and a
:class:`Product` (also all numpy), so shipping it verbatim is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ptychodus.api.reconstructor import ReconstructInput

if TYPE_CHECKING:
    from ptycho_torch.config_params import (
        DataConfig,
        DatagenConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )

__all__ = [
    'ReconstructPayload',
    'TrainPayload',
]


@dataclass(frozen=True)
class ReconstructPayload:
    """Everything the child needs to load a checkpoint and run inference once.

    No configs: the child rebuilds its ``ConfigManager`` from the ones saved
    inside the checkpoint, so ptychodus settings play no part in inference.
    """

    # Which of ('Unsupervised', 'Supervised') this reconstructor targets. The
    # child warns if the checkpoint disagrees.
    model_training_mode: str

    # Path to the checkpoint .ckpt file recorded by the parent-side
    # SubprocessReconstructor.load_model_from_file() call. Required for
    # inference (child raises RuntimeError if None).
    model_path: Path | None

    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    """Everything the child needs to run one training session."""

    data_config: DataConfig
    model_config: ModelConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    datagen_config: DatagenConfig

    input_path: Path
    output_path: Path

    # Comma-separated GPU indices, or '' to inherit. The child exports this as
    # CUDA_VISIBLE_DEVICES before touching the CUDA runtime, then clamps
    # ``training_config.n_devices`` to what actually became visible.
    visible_gpu_indices: str
