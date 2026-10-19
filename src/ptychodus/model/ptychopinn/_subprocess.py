"""Child-side subprocess entry points for the PtychoPINN backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that imports ``ptycho_torch`` or torch, and every such import is
function-local so that even importing this module pulls in neither. The
parent-side ptychodus process never imports it at all.

Two entry points are exposed:

- :func:`run_reconstruct` -- load a trained bundle, run one inference pass,
  stream back a single :class:`ReconstructOutput`.
- :func:`run_train` -- run one training session through the backend's public
  dataset-to-bundle door, then stream back the final :class:`TrainOutput`
  plus the saved bundle path.

Neither entry point reads ptychodus settings. Training receives an
already-translated settings mapping on its payload; inference takes its model,
data and probe identity from the bundle.
"""

from __future__ import annotations

import csv
import logging
import pickle
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.object import Object
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
    ReconstructOutput,
    TrainOutput,
)
from ._payload import ReconstructPayload, TrainPayload

logger = logging.getLogger(__name__)

# Ptychodus exports detector patterns as photon counts and labels them as such,
# so the exported source and the bundle target agree and nothing is rescaled.
_SCALE_CONTRACT_VERSION = 'ci_intensity_v2'
_MEASUREMENT_DOMAIN = 'count_intensity'


def _resolve_device(accelerator: str) -> str:
    """Map the configured accelerator onto a torch device string.

    Runs child-side because deciding between CUDA and CPU means asking torch
    whether a device is actually present, which the parent must not do.
    """
    import torch

    requested = accelerator.strip().casefold()

    if requested == 'cpu':
        return 'cpu'

    if requested == 'mps':
        return 'mps'

    if torch.cuda.is_available():
        return 'cuda'

    if requested in ('gpu', 'cuda'):
        logger.warning(
            'Accelerator %r requested but no CUDA device is available; using CPU.', accelerator
        )

    return 'cpu'


def _build_arrays(parameters: Any) -> dict[str, Any]:
    """Convert a ReconstructInput into the backend's flat-acquisition mapping.

    Mirrors the training-data export so inference sees the same preprocessing
    and the same declared measurement domain the model was trained on.
    """
    object_geometry = parameters.product.object_.get_geometry()
    positions = parameters.product.probe_positions
    xcoords = object_geometry.map_probe_positions_to_object_x_px(positions)
    ycoords = object_geometry.map_probe_positions_to_object_y_px(positions)

    return {
        'xcoords': xcoords,
        'ycoords': ycoords,
        'xcoords_start': xcoords,
        'ycoords_start': ycoords,
        # Repaired the same way exported training data is.
        'diff3d': inpaint_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels),
        'probeGuess': parameters.product.probes.get_probe_no_opr().get_array(),
        'objectGuess': parameters.product.object_.get_layer(0),
        'scan_index': numpy.zeros(len(positions), dtype=int),
        'scale_contract_version': _SCALE_CONTRACT_VERSION,
        'measurement_domain': _MEASUREMENT_DOMAIN,
    }


def run_reconstruct(payload: ReconstructPayload, queue: Queue[Any]) -> None:
    """Child entry point for one inference pass. Streams a single ReconstructOutput."""
    if payload.model_bundle_dir is None:
        raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

    from ptycho_torch.inference import ReconstructionRuntimeParams, reconstruct_from_arrays
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    loaded_model, _bundle_params = load_inference_bundle_torch(payload.model_bundle_dir)

    bundle_mode = getattr(loaded_model.model_config, 'mode', None)
    if bundle_mode is not None and bundle_mode != payload.model_training_mode:
        logger.warning(
            'Loaded bundle mode %r does not match reconstructor mode %r; '
            'predictions may be inconsistent.',
            bundle_mode,
            payload.model_training_mode,
        )

    # The bundle owns geometry and trimming; only the presentation knobs below
    # are ptychodus's to choose at reconstruct time.
    inference_config = replace(
        loaded_model.inference_config,
        patch_weighting=payload.patch_weighting,
        batch_size=payload.inference_batch_size,
    )

    parameters = payload.reconstruct_input
    device = _resolve_device(payload.accelerator)

    with tempfile.TemporaryDirectory(prefix='ptychopinn-reconstruct-') as workspace:
        result = reconstruct_from_arrays(
            loaded_model,
            _build_arrays(parameters),
            runtime_params=ReconstructionRuntimeParams(
                # Replaced during staging from the model and the arrays; the
                # call still requires the fields to be present.
                data_config=loaded_model.data_config,
                training_config=loaded_model.training_config,
                inference_config=inference_config,
                source_metadata={},
                quiet=True,
            ),
            workspace=workspace,
            device=device,
            num_workers=payload.num_workers,
        )
        # Snapshot before the workspace unlinks its memory-mapped backing.
        object_out_array = numpy.squeeze(numpy.asarray(result.complex_canvas))

    object_in = parameters.product.object_
    object_out = Object(
        array=object_out_array,
        layer_spacing_m=object_in.layer_spacing_m,
        pixel_geometry=object_in.get_pixel_geometry(),
        center=object_in.get_center(),
    )

    losses: Sequence[LossValue] = []
    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_out,
        losses=losses,
    )

    queue.put(
        (TAG_OUTPUT, pickle.dumps(ReconstructOutput(product=product, progress=1, progress_goal=1)))
    )


def _latest_metrics_csv(output_path: Path) -> Path | None:
    """Return the newest Lightning CSV metrics file, or None if there is none."""
    candidates: list[tuple[int, Path]] = []

    for version_dir in (output_path / 'lightning_logs').glob('version_*'):
        metrics_path = version_dir / 'metrics.csv'
        if not metrics_path.is_file():
            continue
        try:
            version = int(version_dir.name.removeprefix('version_'))
        except ValueError:
            continue
        candidates.append((version, metrics_path))

    if not candidates:
        return None

    return max(candidates)[1]


def _select_loss_column(field_names: Sequence[str], infix: str) -> str | None:
    """Return the single loss column for one split, or None if it is ambiguous.

    The backend assembles these names from the loss mode and the optional
    auxiliary terms, so they are matched by shape rather than enumerated.
    """
    matches = [name for name in field_names if infix in name and name.endswith('_loss')]

    if len(matches) == 1:
        return matches[0]

    if matches:
        logger.warning('Ambiguous %r loss columns in the training log: %s', infix, matches)
    else:
        logger.warning('No %r loss column found in the training log.', infix)

    return None


def _read_epoch_losses(output_path: Path) -> tuple[list[LossValue], list[LossValue]]:
    """Read per-epoch training and validation losses from the Lightning CSV log.

    The public training door takes no callback, so losses are recovered after
    the run rather than streamed during it. A missing or unreadable log costs
    the loss curves, never the trained model.
    """
    metrics_path = _latest_metrics_csv(output_path)

    if metrics_path is None:
        logger.warning('No Lightning metrics log found under "%s".', output_path)
        return [], []

    with metrics_path.open(newline='') as metrics_file:
        rows = list(csv.DictReader(metrics_file))
        field_names = rows[0].keys() if rows else []

    train_column = _select_loss_column(list(field_names), '_train')
    val_column = _select_loss_column(list(field_names), '_val')

    def collect(column: str | None) -> list[LossValue]:
        if column is None:
            return []

        values: list[LossValue] = []

        for row in rows:
            raw_value = row.get(column)
            raw_epoch = row.get('epoch')

            if not raw_value or raw_epoch is None or raw_epoch == '':
                continue

            try:
                values.append(LossValue(epoch=int(float(raw_epoch)), value=float(raw_value)))
            except ValueError:
                continue

        return values

    return collect(train_column), collect(val_column)


def run_train(payload: TrainPayload, queue: Queue[Any]) -> None:
    """Child entry point for one training session.

    Masks the visible GPUs, then hands the already-translated settings mapping
    to the backend's public dataset-to-bundle training door.
    """

    from ptycho_torch.execution_request import ExecutionRequest
    from ptycho_torch.train import train

    execution_values: dict[str, Any] = {
        'accelerator': payload.accelerator,
        # Resolves to the number of devices actually visible after the mask
        # above, so no manual clamping is needed.
        'devices': 'auto',
        'num_workers': payload.num_workers,
        'enable_progress_bar': False,
    }

    bundle_path = train(
        payload.input_path / 'train_data.npz',
        payload.output_path,
        payload.settings,
        execution_config=ExecutionRequest(
            values=execution_values,
            explicit_fields=frozenset(execution_values),
        ),
    )

    training_loss, validation_loss = _read_epoch_losses(payload.output_path)

    # Training has finished by now, so report it as complete even when the log
    # could not be read -- a run that produced a bundle but no loss curve is
    # still a finished run, and zero progress would read as a failure.
    progress = len(training_loss) or int(payload.settings.get('epochs', 0))

    queue.put(
        (
            TAG_TRAIN_OUTPUT,
            pickle.dumps(
                TrainOutput(
                    training_loss=training_loss,
                    validation_loss=validation_loss,
                    progress=progress,
                    progress_goal=progress,
                )
            ),
        )
    )
    queue.put((TAG_MODEL_SAVED, str(bundle_path)))
