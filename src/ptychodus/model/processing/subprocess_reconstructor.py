"""Generic reconstructor adapter that runs the underlying backend in a subprocess.

Wraps any backend that satisfies the ``Reconstructor`` /
``TrainableReconstructor`` shape (as a child-side entry point) into a
parent-safe object that ptychodus can dispatch to without ever importing a
GPU framework.

The parent side of the adapter has zero GPU imports. Every
``reconstruct()`` / ``train()`` call spawns a fresh ``spawn``-context
subprocess via :mod:`._subprocess_protocol`, streams outputs back, and dies
at end-of-iteration. Consumers of the reconstructor iterator (see
``ReconstructBackgroundTask``) see the same per-iteration ``ReconstructOutput``
they did in the in-process version.

Trainable-model lifecycle
-------------------------

``load_model_from_file`` on the parent-side adapter records the path only --
the fresh inference child does the real load per call. ``save_model`` copies
or archives the loaded-from path to the destination (using a backend-supplied
callback so ptychopinn's zip-a-bundle-dir semantic and ptychopinn_torch's
copy-the-.ckpt semantic can share the same adapter). ``export_training_data``
runs in the parent (all current implementations are pure-numpy and touch no
GPU framework).
"""

from __future__ import annotations

import logging
import pickle
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)

from ._subprocess_protocol import run_subprocess

logger = logging.getLogger(__name__)

__all__ = [
    'SubprocessReconstructor',
]


# Tags emitted by child entry points and consumed by this adapter.
TAG_OUTPUT = 'output'  # payload = pickle.dumps(ReconstructOutput)
TAG_TRAIN_OUTPUT = 'train_output'  # payload = pickle.dumps(TrainOutput)
TAG_MODEL_SAVED = 'model_saved'  # payload = str path where child wrote checkpoint
TAG_SETTINGS_SYNC = 'settings_sync'  # payload = {group_name: {param_name: value_str}}


# Type aliases for the callbacks each backend supplies.
BuildReconstructPayload = Callable[[ReconstructInput, 'Path | None'], Any]
"""(reconstruct_input, loaded_model_path_or_none) -> pickled payload for the child."""

BuildTrainPayload = Callable[[Path, Path], Any]
"""(input_dir, output_dir) -> pickled payload for the child."""

ExportTrainingData = Callable[[Path, ReconstructInput], None]
"""(file_path, reconstruct_input) -> None. Runs parent-side; must not touch GPU."""

SaveModel = Callable[[Path, Path], None]
"""(source_path_loaded_by_child, destination_path) -> None. Runs parent-side."""

ApplySettingsSync = Callable[[dict[str, dict[str, str]]], None]
"""Optional handler for ('settings_sync', dict) messages the child emits."""


class SubprocessReconstructor(TrainableReconstructor):
    """Parent-side adapter that runs any reconstructor in a fresh subprocess.

    Implements the full :class:`TrainableReconstructor` interface. When
    ``is_trainable`` is False, the trainable methods raise
    :class:`NotImplementedError` so the object still slots into ``ProcessingCore``'s
    list of ``Reconstructor`` instances without special-casing.

    The parent-side object holds no GPU state. It stores:
    - the child entry-point strings (dotted module paths),
    - backend-supplied callbacks that build payloads and export training data,
    - the loaded-model path (as recorded by ``load_model_from_file`` /
      remembered from the last ``train`` call).

    Each ``reconstruct()`` / ``train()`` call spawns exactly one child; the
    child dies before the call returns.
    """

    def __init__(
        self,
        *,
        name: str,
        reconstruct_entry_point: str,
        progress_goal_fn: Callable[[], int],
        build_reconstruct_payload: BuildReconstructPayload,
        is_trainable: bool = False,
        train_entry_point: str | None = None,
        build_train_payload: BuildTrainPayload | None = None,
        model_file_filter: str = '',
        model_file_extension: str = '',
        training_data_file_filter: str = '',
        export_training_data: ExportTrainingData | None = None,
        save_model: SaveModel | None = None,
        apply_settings_sync: ApplySettingsSync | None = None,
        terminate_grace_sec: float = 10.0,
    ) -> None:
        super().__init__()
        self._name = name
        self._reconstruct_entry_point = reconstruct_entry_point
        self._progress_goal_fn = progress_goal_fn
        self._build_reconstruct_payload = build_reconstruct_payload
        self._is_trainable = is_trainable
        self._train_entry_point = train_entry_point
        self._build_train_payload = build_train_payload
        self._model_file_filter = model_file_filter
        self._model_file_extension = model_file_extension
        self._training_data_file_filter = training_data_file_filter
        self._export_training_data = export_training_data
        self._save_model_fn = save_model
        self._apply_settings_sync = apply_settings_sync
        self._terminate_grace_sec = terminate_grace_sec

        self._loaded_model_path: Path | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    def get_progress_goal(self) -> int:
        return self._progress_goal_fn()

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        payload = self._build_reconstruct_payload(parameters, self._loaded_model_path)

        with run_subprocess(
            self._reconstruct_entry_point,
            payload,
            terminate_grace_sec=self._terminate_grace_sec,
        ) as events:
            for event in events:
                tag = event[0]
                if tag == TAG_OUTPUT:
                    yield pickle.loads(event[1])
                elif tag == TAG_SETTINGS_SYNC:
                    if self._apply_settings_sync is not None:
                        try:
                            self._apply_settings_sync(event[1])
                        except Exception:
                            logger.exception('Failed to apply settings sync from subprocess.')
                else:
                    logger.debug(
                        f'{self._name}: dropping unrecognized subprocess message tag {tag!r}'
                    )

    def is_model_loaded(self) -> bool:
        return self._loaded_model_path is not None

    def get_model_file_filter(self) -> str:
        return self._model_file_filter

    def load_model_from_file(self, file_path: Path) -> None:
        # Parent-side: just remember the path. The child does the actual load
        # (which touches GPU frameworks) per reconstruct/train call.
        self._loaded_model_path = file_path

    def get_model_file_extension(self) -> str:
        return self._model_file_extension

    def save_model(self, file_path: Path) -> None:
        if self._loaded_model_path is None:
            raise RuntimeError(
                f'Cannot save {self._name} model: no model has been loaded or trained.'
            )
        if self._save_model_fn is not None:
            self._save_model_fn(self._loaded_model_path, file_path)
        else:
            shutil.copyfile(self._loaded_model_path, file_path)

    def get_training_data_file_filter(self) -> str:
        return self._training_data_file_filter

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        if self._export_training_data is None:
            raise NotImplementedError(f'{self._name} does not support exporting training data.')
        self._export_training_data(file_path, parameters)

    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        if not self._is_trainable:
            raise NotImplementedError(f'{self._name} does not support training.')
        if self._train_entry_point is None or self._build_train_payload is None:
            raise NotImplementedError(f'{self._name} does not support training.')

        payload = self._build_train_payload(input_path, output_path)

        with run_subprocess(
            self._train_entry_point,
            payload,
            terminate_grace_sec=self._terminate_grace_sec,
        ) as events:
            for event in events:
                tag = event[0]
                if tag == TAG_TRAIN_OUTPUT:
                    yield pickle.loads(event[1])
                elif tag == TAG_MODEL_SAVED:
                    self._loaded_model_path = Path(event[1])
                elif tag == TAG_SETTINGS_SYNC:
                    if self._apply_settings_sync is not None:
                        try:
                            self._apply_settings_sync(event[1])
                        except Exception:
                            logger.exception('Failed to apply settings sync from subprocess.')
                else:
                    logger.debug(
                        f'{self._name}: dropping unrecognized subprocess message tag {tag!r}'
                    )
