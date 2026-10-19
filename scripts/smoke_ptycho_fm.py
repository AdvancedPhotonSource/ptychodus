"""End-to-end smoke test for the PtychoFM backend against the current ptycho-fm.

Not part of the test suite -- ptycho-fm is not installed in the venv, so this is
run manually with PYTHONPATH pointing at the sibling checkout:

    PYTHONPATH=/home/beams0/SHENKE/Ptychography/ptycho-fm \
        uv run --no-sync python scripts/smoke_ptycho_fm.py [--cpu]

Drives the child entry points directly (no GUI, no subprocess) so failures
surface as ordinary tracebacks. Covers both training objectives, each
``coupled_decoder_mode``, the diffraction-space losses and the probe-envelope
call contract -- everything that needs a real ptycho_fm to prove.
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
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
    ReconstructInput,
)
from ptychodus.api.settings import SettingsRegistry
from ptychodus.plugins.ptycho_fm._payload import ReconstructPayload, TrainPayload
from ptychodus.plugins.ptycho_fm._subprocess import (
    _build_criterion,
    _compute_loss,
    run_reconstruct,
    run_train,
)
from ptychodus.plugins.ptycho_fm.reconstructor import _build_config, build_reconstructor
from ptychodus.plugins.ptycho_fm.settings import (
    PtychoFMDataSettings,
    PtychoFMInferenceSettings,
    PtychoFMModelSettings,
    PtychoFMTrainingSettings,
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
    # Pinned so the baseline phases cover the uncoupled path; the mode sweep
    # below is the only place that varies it.
    model.coupled_decoder_mode.set_value('none')

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
        photon_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
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
        probes=ProbeEnsemble(array=probe_array, opr_weights=None, pixel_geometry=pixel_geometry),
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
            reconstructor.export_training_data(
                stem, _reconstruct_input(IMG_SIZE // 2), options=reconstructor.get_default_options()
            )
        except ValueError as exc:
            print(f'  [ok] export rejected undersized patterns: {exc}')
        else:
            raise AssertionError('export_training_data accepted a wrongly-sized product')

        reconstructor.export_training_data(
            stem, _reconstruct_input(IMG_SIZE), options=reconstructor.get_default_options()
        )
        produced = sorted(p.name for p in Path(tmp).iterdir())
        assert produced == ['scan_dp.hdf5', 'scan_para.hdf5'], produced
        print(f'  [ok] export wrote {produced}')


def run_training(
    mode: str, data: Any, model: Any, training: Any, inference: Any, out_dir: Path
) -> tuple[Path, list[float]]:
    reconstructor = build_reconstructor(mode, data, model, inference, training)
    in_dir = out_dir / 'input'
    in_dir.mkdir(parents=True, exist_ok=True)
    reconstructor.export_training_data(
        in_dir / 'scan', _reconstruct_input(IMG_SIZE), options=reconstructor.get_default_options()
    )

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


def check_coupled_decoder_modes(data: Any, model: Any, training: Any, inference: Any) -> None:
    """Train and infer under each decoder mode, and prove the mode is honoured.

    An inference-only model ignores the probe entirely, so a mode that silently
    collapsed to the uncoupled class would still produce a plausible object.
    Differing loss traces are what rules that out.
    """
    training.loss_function.set_value('mse')
    original_mode = model.coupled_decoder_mode.get_value()
    losses_by_mode: dict[str, list[float]] = {}

    for coupled_decoder_mode in ('none', 'real_imag', 'amp_phase'):
        model.coupled_decoder_mode.set_value(coupled_decoder_mode)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint, losses = run_training(
                'Unsupervised', data, model, training, inference, Path(tmp) / coupled_decoder_mode
            )
            losses_by_mode[coupled_decoder_mode] = losses
            run_inference(checkpoint, data, model, training, inference)

    distinct = {tuple(losses) for losses in losses_by_mode.values()}
    assert len(distinct) == len(losses_by_mode), (
        f'decoder modes produced identical losses ({losses_by_mode}); the mode is not honoured'
    )
    print('  [ok] the three decoder modes optimize distinct architectures')
    model.coupled_decoder_mode.set_value(original_mode)


def check_mode_mismatch_is_reported(data: Any, model: Any, training: Any, inference: Any) -> None:
    """A checkpoint loaded under the wrong mode must name the mode, not the keys."""
    training.loss_function.set_value('mse')
    original_mode = model.coupled_decoder_mode.get_value()
    model.coupled_decoder_mode.set_value('amp_phase')

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint, _ = run_training(
            'Unsupervised', data, model, training, inference, Path(tmp) / 'mismatch'
        )

        model.coupled_decoder_mode.set_value('none')
        payload = ReconstructPayload(
            name='Unsupervised',
            config=_build_config('Unsupervised', data, model, training, inference),
            model_path=checkpoint,
            reconstruct_input=_reconstruct_input(IMG_SIZE),
        )
        q: queue.Queue[Any] = queue.Queue()

        try:
            run_reconstruct(payload, q)  # type: ignore[arg-type]
        except ValueError as exc:
            assert 'coupled_decoder_mode' in str(exc), f'unhelpful message: {exc}'
            print(f'  [ok] mismatch reported: {str(exc).splitlines()[0][:80]}...')
        else:
            raise AssertionError('loading an amp_phase checkpoint as uncoupled did not raise')

    model.coupled_decoder_mode.set_value(original_mode)


def check_new_losses(data: Any, model: Any, training: Any, inference: Any) -> None:
    """Each diffraction-space loss trains to a finite value."""
    original_mode = model.coupled_decoder_mode.get_value()
    model.coupled_decoder_mode.set_value('none')

    for loss_function in ('total_flux', 'probe_aware', 'q_dependent'):
        training.loss_function.set_value(loss_function)

        with tempfile.TemporaryDirectory() as tmp:
            _, losses = run_training(
                'Unsupervised', data, model, training, inference, Path(tmp) / loss_function
            )

        assert all(numpy.isfinite(value) for value in losses), f'{loss_function}: {losses}'
        print(f'  [ok] {loss_function}: finite losses {losses}')

    training.loss_function.set_value('mse')
    model.coupled_decoder_mode.set_value(original_mode)


def check_probe_envelope_dispatch(data: Any, model: Any, training: Any, inference: Any) -> None:
    """The probe-aware loss must actually receive the probe.

    Called with prediction and target alone it still returns a number -- a
    different one -- so only comparing the two call forms catches a regression.
    """
    import torch

    training.loss_function.set_value('probe_aware')
    config = _build_config('Unsupervised', data, model, training, inference)
    criterion = _build_criterion(config['training'])

    assert getattr(criterion, 'uses_probe_envelope', False), (
        'ProbeAwareLoss no longer advertises uses_probe_envelope; the dispatch is dead'
    )

    generator = torch.Generator().manual_seed(0)
    shape = (2, 1, IMG_SIZE, IMG_SIZE)
    prediction = torch.rand(shape, generator=generator)
    target = torch.rand(shape, generator=generator)
    probe = torch.view_as_real(
        torch.randn(2, 1, 2, IMG_SIZE, IMG_SIZE, generator=generator, dtype=torch.complex64)
    )
    normalization = torch.ones(2)
    scale = torch.full((2,), 10000.0)

    with_envelope = _compute_loss(
        criterion, prediction, target, probe=probe, normalization=normalization, scale=scale
    )
    without_envelope = criterion(prediction, target)

    assert not torch.isclose(with_envelope, without_envelope), (
        'the probe envelope changed nothing; _compute_loss is dropping it'
    )
    print(
        f'  [ok] envelope honoured: with={with_envelope.item():.6g}, '
        f'without={without_envelope.item():.6g}'
    )

    training.loss_function.set_value('mse')


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

    print('4. coupled decoder modes')
    check_coupled_decoder_modes(data, model, training, inference)

    print('5. checkpoint/mode mismatch')
    check_mode_mismatch_is_reported(data, model, training, inference)

    print('6. diffraction-space losses')
    check_new_losses(data, model, training, inference)

    print('7. probe-envelope dispatch')
    check_probe_envelope_dispatch(data, model, training, inference)

    print('\nAll smoke checks passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
