"""End-to-end smoke test for the PtychoFM backend against the current ptycho-fm.

Not part of the test suite -- ptycho-fm is not installed in the venv, so this is
run manually with PYTHONPATH pointing at the sibling checkout:

    PYTHONPATH=/home/beams0/SHENKE/Ptychography/ptycho-fm \
        uv run --no-sync python scripts/smoke_ptycho_fm.py [--cpu]

Drives the child entry points directly (no GUI, no subprocess) so failures
surface as ordinary tracebacks.
"""

from __future__ import annotations

import argparse
import pickle
import queue
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.ptycho_fm._payload import ReconstructPayload, TrainPayload
from ptychodus.model.ptycho_fm._subprocess import run_reconstruct, run_train
from ptychodus.model.ptycho_fm.reconstructor import _build_config, build_reconstructor
from ptychodus.model.ptycho_fm.settings import (
    PtychoFMDataSettings,
    PtychoFMInferenceSettings,
    PtychoFMModelSettings,
    PtychoFMTrainingSettings,
)
from ptychodus.model.processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)

# Small enough to train in seconds. The decoder upsamples the token grid by
# 2**num_stages, so (img_size / patch_size) * 2**num_stages must come back to
# img_size: (64 / 8) * 2**3 = 64.
IMG_SIZE = 64
PATCH_SIZE = 8
NUM_STAGES = 3
N_PATTERNS = 12
OBJECT_SIZE = 96


def _settings() -> tuple[Any, Any, Any, Any]:
    registry = SettingsRegistry()
    data = PtychoFMDataSettings(registry)
    model = PtychoFMModelSettings(registry)
    training = PtychoFMTrainingSettings(registry)
    inference = PtychoFMInferenceSettings(registry)

    model.img_size.set_value(IMG_SIZE)
    model.patch_size.set_value(PATCH_SIZE)
    model.embed_dim.set_value(64)
    model.depth.set_value(2)
    model.num_heads.set_value(2)
    model.decoder_base_channels.set_value(8)
    model.decoder_latent_dim.set_value(64)
    model.decoder_num_stages.set_value(NUM_STAGES)

    data.max_probe_modes.set_value(2)
    data.num_workers.set_value(0)

    training.epochs.set_value(2)
    training.batch_size.set_value(4)
    training.loss_function.set_value('mse')

    inference.batch_size.set_value(4)
    inference.central_crop.set_value(8)
    inference.pad.set_value(4)
    return data, model, training, inference


def _product(pattern_size: int) -> Product:
    rng = numpy.random.default_rng(0)
    metadata = ProductMetadata(
        name='smoke',
        comments='',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
    )
    pixel_geometry = PixelGeometry(1.0e-9, 1.0e-9)

    span = (OBJECT_SIZE - pattern_size) * 1.0e-9
    positions = [
        ProbePosition(
            index=i,
            x_m=(i % 4) / 3.0 * span - span / 2.0,
            y_m=(i // 4) / 2.0 * span - span / 2.0,
        )
        for i in range(N_PATTERNS)
    ]

    object_array = (
        rng.normal(1.0, 0.05, (1, OBJECT_SIZE, OBJECT_SIZE))
        + 1j * rng.normal(0.0, 0.05, (1, OBJECT_SIZE, OBJECT_SIZE))
    ).astype(numpy.complex64)

    probe_array = numpy.zeros((1, 2, pattern_size, pattern_size), dtype=numpy.complex64)
    probe_array[0, 0] = 1.0

    return Product(
        metadata=metadata,
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeSequence(array=probe_array, opr_weights=None, pixel_geometry=pixel_geometry),
        object_=Object(
            array=object_array,
            layer_spacing_m=[],
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(0.0, 0.0),
        ),
        losses=[],
    )


def _reconstruct_input(pattern_size: int) -> ReconstructInput:
    rng = numpy.random.default_rng(1)
    patterns = rng.poisson(50.0, (N_PATTERNS, pattern_size, pattern_size)).astype(numpy.float32)
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=numpy.zeros((pattern_size, pattern_size), dtype=numpy.bool_),
        product=_product(pattern_size),
    )


def _drain(q: queue.Queue[Any]) -> list[Any]:
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def check_export_validation(data: Any, model: Any, training: Any, inference: Any) -> None:
    """The exporter must refuse a product whose patterns are not img_size square."""
    reconstructor = build_reconstructor('Supervised', data, model, inference, training)

    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / 'scan'
        try:
            reconstructor.export_training_data(stem, _reconstruct_input(IMG_SIZE // 2))
        except ValueError as exc:
            print(f'  [ok] export rejected undersized patterns: {exc}')
        else:
            raise AssertionError('export_training_data accepted a wrongly-sized product')

        reconstructor.export_training_data(stem, _reconstruct_input(IMG_SIZE))
        produced = sorted(p.name for p in Path(tmp).iterdir())
        assert produced == ['scan_dp.hdf5', 'scan_para.hdf5'], produced
        print(f'  [ok] export wrote {produced}')


def run_training(
    mode: str, data: Any, model: Any, training: Any, inference: Any, out_dir: Path
) -> tuple[Path, list[float]]:
    reconstructor = build_reconstructor(mode, data, model, inference, training)
    in_dir = out_dir / 'input'
    in_dir.mkdir(parents=True, exist_ok=True)
    reconstructor.export_training_data(in_dir / 'scan', _reconstruct_input(IMG_SIZE))

    model_dir = out_dir / 'output'
    payload = TrainPayload(
        name=mode,
        config=_build_config(mode, data, model, training, inference),
        input_path=in_dir,
        output_path=model_dir,
    )
    q: queue.Queue[Any] = queue.Queue()
    run_train(payload, q)  # type: ignore[arg-type]

    events = _drain(q)
    train_outputs = [pickle.loads(e[1]) for e in events if e[0] == TAG_TRAIN_OUTPUT]
    saved = [e[1] for e in events if e[0] == TAG_MODEL_SAVED]

    assert len(train_outputs) == 2, f'expected 2 TrainOutput, got {len(train_outputs)}'
    assert len(saved) == 1, f'expected 1 TAG_MODEL_SAVED, got {len(saved)}'
    assert Path(saved[0]).is_file(), saved[0]

    losses = [lv.value for lv in train_outputs[-1].training_loss]
    print(f'  [ok] {mode}: 2 epochs, training loss {losses}, checkpoint {Path(saved[0]).name}')
    return Path(saved[0]), losses


def run_inference(checkpoint: Path, data: Any, model: Any, training: Any, inference: Any) -> None:
    parameters = _reconstruct_input(IMG_SIZE)
    payload = ReconstructPayload(
        name='Supervised',
        config=_build_config('Supervised', data, model, training, inference),
        model_path=checkpoint,
        reconstruct_input=parameters,
    )
    q: queue.Queue[Any] = queue.Queue()
    run_reconstruct(payload, q)  # type: ignore[arg-type]

    events = _drain(q)
    outputs = [pickle.loads(e[1]) for e in events if e[0] == TAG_OUTPUT]
    assert len(outputs) == 1, f'expected 1 ReconstructOutput, got {len(outputs)}'

    array = outputs[0].product.object_.get_array()
    expected = parameters.product.object_.get_array().shape
    assert array.shape == expected, f'{array.shape} != {expected}'
    assert numpy.all(numpy.isfinite(array)), 'object contains non-finite values'
    assert numpy.any(array != 0.0), 'object is entirely zero'
    print(
        f'  [ok] inference: progress={outputs[0].progress}, shape={array.shape}, '
        f'|obj| range [{numpy.abs(array).min():.4g}, {numpy.abs(array).max():.4g}]'
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='hide CUDA devices')
    args = parser.parse_args()

    if args.cpu:
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    import torch

    print(f'torch {torch.__version__}, cuda available: {torch.cuda.is_available()}')

    data, model, training, inference = _settings()

    print('1. export validation')
    check_export_validation(data, model, training, inference)

    losses_by_mode = {}
    with tempfile.TemporaryDirectory() as tmp:
        print('2. training')
        for mode in ('Supervised', 'Unsupervised'):
            checkpoint, losses = run_training(
                mode, data, model, training, inference, Path(tmp) / mode
            )
            losses_by_mode[mode] = losses
            if mode == 'Supervised':
                supervised_checkpoint = checkpoint

        assert losses_by_mode['Supervised'] != losses_by_mode['Unsupervised'], (
            'supervised and unsupervised produced identical losses; mode is not honoured'
        )
        print('  [ok] the two modes optimize different objectives')

        print('3. inference')
        run_inference(supervised_checkpoint, data, model, training, inference)

    print('\nAll smoke checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
