"""Payload dataclasses for the PtychoPINN (TensorFlow) subprocess entry points.

``ptycho`` is an optional extra, so it is imported under
:data:`~typing.TYPE_CHECKING` only: this module has ``from __future__ import
annotations`` and both payloads are dataclasses, whose field annotations are
never evaluated. That keeps ``ptychodus.model`` importable without the extra
installed. The factory defers the matching runtime imports into its config
builders.

Only ``ptycho.config.config`` may be reached from the parent: it pulls in just
``dataclasses``, ``enum``, and ``typing`` -- no TensorFlow, as evidenced by the
fact that it imports cleanly in an environment where TensorFlow is absent
entirely. Anything else under ``ptycho.*`` (``raw_data``, ``probe``,
``tf_helper``, ``loader``) needs TensorFlow and must stay inside the child; see
[tests/test_no_gpu_context.py](../../../../tests/test_no_gpu_context.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ptychodus.api.reconstruct import ReconstructInput

if TYPE_CHECKING:
    from ptycho.config.config import InferenceConfig, TrainingConfig

__all__ = [
    'MODEL_FILE_NAME',
    'ReconstructPayload',
    'TrainPayload',
]


MODEL_FILE_NAME: Final[str] = 'wts.h5.zip'
"""Name ``ptycho.model_manager`` gives the weights archive inside a bundle."""


@dataclass(frozen=True)
class ReconstructPayload:
    """Everything the child needs to load a bundle and run inference once."""

    # Fully populated by the parent, including the nested ModelConfig whose
    # ``N`` is the (already validated square) diffraction pattern size.
    inference_config: InferenceConfig

    # Directory the child loads the model from. The parent has already
    # unpacked an outer .zip if there was one.
    model_bundle_dir: Path

    # Arguments to RawData.generate_grouped_data, which only the child can call.
    n_nearest_neighbors: int
    n_samples: int

    reconstruct_input: ReconstructInput


@dataclass(frozen=True)
class TrainPayload:
    """Everything the child needs to run one training session.

    ``training_config.model.N`` is left at its default: the pattern size is
    known only once the child loads ``train_data.npz``, so the child fills it
    in before handing the config to ``run_cdi_example``.
    """

    training_config: TrainingConfig

    input_path: Path
    output_path: Path
