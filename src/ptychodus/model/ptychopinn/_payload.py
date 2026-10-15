"""Payload dataclasses for the PtychoPINN subprocess entry points.

Nothing here imports ``ptycho_torch``, and nothing here imports torch. Both
payloads carry only built-in scalars, strings, paths and a mapping, so the
parent process never pays an import cost and never risks acquiring a GPU
context while assembling one. The child turns the mapping into resolved
backend configuration; see :mod:`._subprocess`.

Everything here is picklable: :class:`ReconstructInput` is a frozen dataclass
of numpy arrays and a :class:`Product` (also all numpy), so shipping it
verbatim is correct.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptychodus.api.reconstruct import ReconstructInput

__all__ = [
    'ReconstructPayload',
    'TrainPayload',
]


@dataclass(frozen=True)
class ReconstructPayload:
    """Everything the child needs to load a bundle and run inference once.

    The bundle owns the model, data and probe identity, so ptychodus supplies
    only the runtime knobs that the backend treats as presentation choices
    rather than sealed identity.
    """

    # Which of ('Unsupervised', 'Supervised') this reconstructor targets. The
    # child warns if the loaded bundle disagrees.
    model_training_mode: str

    # Directory holding ``wts.h5.zip``, resolved parent-side from the path the
    # user loaded. Required for inference (child raises RuntimeError if None).
    model_bundle_dir: Path | None

    reconstruct_input: ReconstructInput

    patch_weighting: str
    inference_batch_size: int

    # Resolved to a torch device string child-side, where CUDA availability can
    # actually be observed.
    accelerator: str
    num_workers: int


@dataclass(frozen=True)
class TrainPayload:
    """Everything the child needs to run one training session."""

    # Backend resolver names to values, as documented in the backend's
    # configuration reference. Deliberately excludes ``nphotons``; see
    # ``reconstructor._build_settings``.
    settings: Mapping[str, Any]

    input_path: Path
    output_path: Path

    # Comma-separated GPU indices, or '' to inherit. The child exports this as
    # CUDA_VISIBLE_DEVICES before touching the CUDA runtime.
    visible_gpu_indices: str

    accelerator: str
    num_workers: int
