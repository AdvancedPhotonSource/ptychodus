"""Payload dataclasses for the PtychoPINN-Torch subprocess entry points.

Parent-safe: no torch, no lightning, no ptycho_torch imports. Every field is
picklable and made of numpy arrays and plain scalars, so the whole payload
survives the pickle round-trip that ``multiprocessing.get_context('spawn')``
requires. :class:`ReconstructInput` itself is a frozen dataclass of numpy
arrays and a :class:`Product` (also all numpy), so shipping it verbatim is
correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptychodus.api.reconstructor import ReconstructInput


@dataclass(frozen=True)
class ReconstructPayload:
    """Everything the child needs to load a checkpoint and run inference once."""

    # Which of ('Unsupervised', 'Supervised') this reconstructor targets.
    model_training_mode: str

    # Path to the checkpoint .ckpt file recorded by the parent-side
    # SubprocessReconstructor.load_model_from_file() call. Required for
    # inference (child raises RuntimeError if None).
    model_path: Path | None

    is_developer_mode_enabled: bool

    # Serialized settings — the child rehydrates a SettingsRegistry from this
    # string via _subprocess_protocol.load_settings_registry_from_string.
    settings_ini: str

    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    """Everything the child needs to run one training session."""

    model_training_mode: str
    is_developer_mode_enabled: bool
    settings_ini: str

    input_path: Path
    output_path: Path

    # DDP knobs. The child sets CUDA_VISIBLE_DEVICES from
    # ``visible_gpu_indices`` BEFORE importing torch, then instantiates
    # ``Trainer(strategy=distributed_strategy, devices=n_devices)``.
    n_devices: int
    distributed_strategy: str
    visible_gpu_indices: str  # comma-separated indices, or '' to inherit
