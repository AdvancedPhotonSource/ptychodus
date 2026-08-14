"""Payload dataclasses for the PtychoNN subprocess entry points.

Parent-safe: no ptychonn / torch / lightning imports.

The subprocess carries a small pydantic ``BaseModel`` (rather than a serialized
ptychodus ``SettingsRegistry``) so the child does not need to import
``PtychoNNModelSettings`` / ``PtychoNNTrainingSettings`` or rehydrate a
registry just to read a handful of scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ptychodus.api.reconstruct import ReconstructInput


class PtychoNNReconstructConfig(BaseModel):
    """Minimal scalar config the child needs to build a ``LitReconSmallModel``."""

    model_config = ConfigDict(frozen=True)

    enable_amplitude: bool
    num_convolution_kernels: int
    use_batch_normalization: bool
    max_learning_rate: float
    min_learning_rate: float


class PtychoNNTrainConfig(PtychoNNReconstructConfig):
    """Adds the four train-only scalars ``ptychonn.train`` reads."""

    batch_size: int
    training_epochs: int
    status_interval_in_epochs: int
    validation_set_fractional_size: float


@dataclass(frozen=True)
class ReconstructPayload:
    config: PtychoNNReconstructConfig
    model_path: Path | None
    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    config: PtychoNNTrainConfig
    input_path: Path
    output_path: Path
