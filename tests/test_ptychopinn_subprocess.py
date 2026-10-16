"""Child-side contract for the PtychoPINN backend.

The parent translates settings into plain values and the child turns them into
backend calls. These tests pin the shape of those calls against a stubbed
backend, because the real one needs lightning and a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.spawn import ChildError
from ptychodus.api.reconstruct import SubprocessReconstructor
from ptychodus.model.ptychopinn._payload import ReconstructPayload, TrainPayload
from ptychodus.model.ptychopinn._subprocess import _build_arrays, _read_epoch_losses

_TESTS_DIR = str(Path(__file__).parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

FIXTURES_MODULE = 'subprocess_child_fixtures'

NUM_POSITIONS = 4
DETECTOR_PX = 16
NUM_MODES = 3


def _make_reconstruct_input() -> ReconstructInput:
    rng = numpy.random.default_rng(5)
    pixel_geometry = PixelGeometry(width_m=1.0e-8, height_m=1.0e-8)
    probe_shape = (2, NUM_MODES, DETECTOR_PX, DETECTOR_PX)
    object_shape = (1, 32, 32)

    product = Product(
        metadata=ProductMetadata(
            name='fixture',
            comments='',
            detector_distance_m=2.0,
            photon_energy_eV=10000.0,
            probe_photon_count=0.0,
            exposure_time_s=0.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(
            [ProbePosition(i, i * 1.0e-8, i * 2.0e-8) for i in range(NUM_POSITIONS)]
        ),
        probes=ProbeEnsemble(
            array=rng.standard_normal(probe_shape) + 1j * rng.standard_normal(probe_shape),
            opr_weights=rng.standard_normal((NUM_POSITIONS, 2)),
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=rng.standard_normal(object_shape) + 1j * rng.standard_normal(object_shape),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
            layer_spacing_m=[],
        ),
        losses=[],
    )

    patterns = rng.integers(0, 1000, size=(NUM_POSITIONS, DETECTOR_PX, DETECTOR_PX)).astype(
        numpy.uint16
    )
    bad_pixels = numpy.zeros((DETECTOR_PX, DETECTOR_PX), dtype=bool)
    bad_pixels[4:7, 4:7] = True

    return ReconstructInput(patterns, bad_pixels, product)


def _captured(excinfo: pytest.ExceptionInfo[ChildError]) -> dict[str, Any]:
    error = excinfo.value
    assert error.child_exception_type == 'ChildCallCapturedError'
    assert error.child_exception is not None
    captured = error.child_exception.args[0]
    assert isinstance(captured, dict)
    return captured


# ----------------------------- spawned child -----------------------------


def test_train_hands_the_public_door_a_settings_mapping() -> None:
    """Training goes through the dataset-to-bundle door, not a config object."""
    payload = TrainPayload(
        settings={'mode': 'Unsupervised', 'epochs': 3},
        input_path=Path('/unused/input'),
        output_path=Path('/unused/output'),
        visible_gpu_indices='',
        accelerator='cpu',
        num_workers=2,
    )
    adapter = SubprocessReconstructor(
        name='PtychoPINN train probe',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:capture_ptychopinn_train_call',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=lambda _parameters, _loaded: payload,
        is_trainable=True,
        train_entry_point=f'{FIXTURES_MODULE}:capture_ptychopinn_train_call',
        build_train_payload=lambda _input, _output: payload,
    )

    with pytest.raises(ChildError) as excinfo:
        list(
            adapter.train(
                Path('/unused/input'), Path('/unused/output'), options=adapter.get_default_options()
            )
        )

    captured = _captured(excinfo)
    assert captured['dataset_name'] == 'train_data.npz'
    assert captured['settings_is_mapping'] is True
    assert captured['mode'] == 'Unsupervised'
    # Naming nphotons would rescale already-counted data; see _build_settings.
    assert captured['has_nphotons'] is False
    assert captured['execution_values']['devices'] == 'auto'
    assert captured['execution_values']['accelerator'] == 'cpu'
    assert captured['execution_values']['num_workers'] == 2
    # Every supplied runtime value must be declared explicit, or the factory
    # treats it as a default it may discard.
    assert captured['explicit_fields'] == sorted(captured['execution_values'])


def test_reconstruct_stages_declared_arrays_through_the_embedder_seam() -> None:
    payload = ReconstructPayload(
        model_training_mode='Unsupervised',
        model_bundle_dir=Path('/unused'),
        reconstruct_input=_make_reconstruct_input(),
        patch_weighting='uniform',
        inference_batch_size=64,
        accelerator='auto',
        num_workers=3,
    )
    adapter = SubprocessReconstructor(
        name='PtychoPINN reconstruct probe',
        reconstruct_entry_point=f'{FIXTURES_MODULE}:capture_ptychopinn_reconstruct_call',
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=lambda _parameters, _loaded: payload,
    )

    with pytest.raises(ChildError) as excinfo:
        list(adapter.reconstruct(SimpleNamespace(), options=adapter.get_default_options()))  # type: ignore[arg-type]

    captured = _captured(excinfo)
    assert captured['array_keys'] == sorted(
        [
            'diff3d',
            'measurement_domain',
            'objectGuess',
            'probeGuess',
            'scale_contract_version',
            'scan_index',
            'xcoords',
            'xcoords_start',
            'ycoords',
            'ycoords_start',
        ]
    )
    # The bundle owns geometry; these two are ptychodus's to choose.
    assert captured['patch_weighting'] == 'uniform'
    assert captured['batch_size'] == 64
    # 'auto' with no CUDA device present resolves to the CPU.
    assert captured['device'] == 'cpu'
    assert captured['num_workers'] == 3


# ------------------------------ _build_arrays ------------------------------
#
# Inference must preprocess and describe its input exactly as the exported
# training data was, or the model sees something it never trained on.


def test_build_arrays_passes_every_probe_mode() -> None:
    parameters = _make_reconstruct_input()

    probe_guess = _build_arrays(parameters)['probeGuess']

    assert probe_guess.shape == (NUM_MODES, DETECTOR_PX, DETECTOR_PX)
    numpy.testing.assert_array_equal(probe_guess, parameters.product.probes.get_array()[0])


def test_build_arrays_repairs_bad_pixels() -> None:
    """Masked positions are inpainted rather than handed over as raw artifacts."""
    parameters = _make_reconstruct_input()
    bad_pixels = parameters.bad_pixels
    assert bad_pixels is not None

    diff3d = _build_arrays(parameters)['diff3d']
    expected = inpaint_bad_pixels(parameters.diffraction_patterns, bad_pixels)

    numpy.testing.assert_array_equal(diff3d, expected)
    assert diff3d.dtype == parameters.diffraction_patterns.dtype
    # The repair has to actually change something, or this proves nothing.
    assert not numpy.array_equal(
        diff3d[:, bad_pixels], parameters.diffraction_patterns[:, bad_pixels]
    )


def test_build_arrays_declares_the_same_contract_as_the_training_export() -> None:
    arrays = _build_arrays(_make_reconstruct_input())

    assert arrays['scale_contract_version'] == 'ci_intensity_v2'
    assert arrays['measurement_domain'] == 'count_intensity'


# ---------------------------- _read_epoch_losses ----------------------------
#
# The public training door takes no callback, so losses are recovered from the
# CSV logger afterwards. The column names vary with the configured loss, so
# they are matched by shape rather than enumerated.


def _write_metrics_csv(output_path: Path, version: int, body: str) -> None:
    version_dir = output_path / 'lightning_logs' / f'version_{version}'
    version_dir.mkdir(parents=True)
    (version_dir / 'metrics.csv').write_text(body)


def test_read_epoch_losses_matches_columns_by_shape(tmp_path: Path) -> None:
    _write_metrics_csv(
        tmp_path,
        0,
        'epoch,poisson_train_Amp_loss,poisson_val_Amp_loss\n0,1.5,2.5\n1,0.5,1.5\n',
    )

    training, validation = _read_epoch_losses(tmp_path)

    assert [(loss.epoch, loss.value) for loss in training] == [(0, 1.5), (1, 0.5)]
    assert [(loss.epoch, loss.value) for loss in validation] == [(0, 2.5), (1, 1.5)]


def test_read_epoch_losses_prefers_the_newest_version(tmp_path: Path) -> None:
    _write_metrics_csv(tmp_path, 0, 'epoch,mae_train_loss\n0,9.0\n')
    _write_metrics_csv(tmp_path, 10, 'epoch,mae_train_loss\n0,1.0\n')

    training, _ = _read_epoch_losses(tmp_path)

    assert [loss.value for loss in training] == [1.0]


def test_read_epoch_losses_ignores_non_loss_metrics(tmp_path: Path) -> None:
    """Only the loss columns are losses; the alpha/grad-norm stats are not."""
    _write_metrics_csv(
        tmp_path,
        0,
        'epoch,mae_pred_l2_alpha_mean_train,grad_norm_preclip_step,mae_train_loss\n0,7.0,8.0,1.0\n',
    )

    training, _ = _read_epoch_losses(tmp_path)

    assert [loss.value for loss in training] == [1.0]


def test_read_epoch_losses_skips_rows_without_a_value(tmp_path: Path) -> None:
    """Lightning writes a row per logging event, not one per epoch."""
    _write_metrics_csv(tmp_path, 0, 'epoch,mae_train_loss,mae_val_loss\n0,,3.0\n0,1.0,\n')

    training, validation = _read_epoch_losses(tmp_path)

    assert [loss.value for loss in training] == [1.0]
    assert [loss.value for loss in validation] == [3.0]


def test_read_epoch_losses_survives_a_missing_log(tmp_path: Path) -> None:
    """A lost loss curve must never cost the trained model."""
    assert _read_epoch_losses(tmp_path) == ([], [])
