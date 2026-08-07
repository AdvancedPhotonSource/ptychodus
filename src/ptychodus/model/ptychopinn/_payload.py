"""Payload dataclasses for the PtychoPINN (TensorFlow) subprocess entry points.

Parent-safe: no tensorflow, no ptycho imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ptychodus.api.reconstructor import ReconstructInput


@dataclass(frozen=True)
class ReconstructPayload:
    name: str  # 'PINN' or 'Supervised'
    model_bundle_path: Path | None  # zip file or bundle dir path recorded by parent
    is_developer_mode_enabled: bool
    settings_ini: str
    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    name: str
    is_developer_mode_enabled: bool
    settings_ini: str
    input_path: Path
    output_path: Path
