"""Payload dataclasses for the PtychoFM (ptycho-vit) subprocess entry points.

``ptycho_vit`` is an optional extra, so nothing in this module imports it. The
config the child needs is carried as a plain nested ``dict[str, Any]`` -- the
same shape ptycho_vit's own ``config.yaml`` produces -- assembled parent-side
by the factory. Dicts are picklable and pull in no framework, so the parent
never touches torch just to build a payload.

The training mode (``'Unsupervised'`` / ``'Supervised'``) is carried in the
``name`` field, matching PtychoPINN's ``model_type`` convention. ptycho_vit
``.pth`` files store a bare ``state_dict`` with no mode metadata, so no
checkpoint-vs-reconstructor mode check is possible today.
"""

from __future__ import annotations

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
    """Everything the child needs to load a checkpoint and run inference once."""

    # 'Unsupervised' or 'Supervised' — the reconstructor's display name.
    name: str

    # ptycho_vit config as a nested dict (data/model/training/inference sections),
    # matching config.yaml. Assembled parent-side from the settings groups.
    config: dict[str, Any]

    # Path to the .pth weights file recorded by
    # SubprocessReconstructor.load_model_from_file() parent-side. Required for
    # inference; the child raises RuntimeError if None.
    model_path: Path | None

    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    """Everything the child needs to run one training session.

    Ptycho-vit training reads its dataset from a directory of ``.hdf5`` files,
    so ``input_path`` should be a directory (not a single file). ``output_path``
    is where the child writes ``best.pth`` and any per-epoch snapshots.
    """

    name: str
    config: dict[str, Any]

    input_path: Path
    output_path: Path
