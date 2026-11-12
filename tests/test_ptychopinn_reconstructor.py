"""What the PtychoPINN reconstructor hands its backend.

The reconstructor translates declared options into plain values and the backend
turns them into calls. These tests pin the shape of those calls against a stubbed
backend, because the real one needs lightning and a GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput, TrainOutput
from ptychodus.plugins.ptychopinn.options import (
    Accelerator,
    PatchWeighting,
    PtychoPINNOptions,
    default_options,
)
from ptychodus.plugins.ptychopinn.reconstructor import (
    PtychoPINNReconstructor,
    _build_arrays,
    _read_epoch_losses,
)

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


# ----------------------------- stubbed backend -----------------------------


def _fake_module(name: str, **attrs: Any) -> ModuleType:
    """A stand-in module carrying `attrs`, for faking an absent optional package."""
    module = ModuleType(name)

    # Via setattr because a ModuleType has no declared attributes.
    for attr, value in attrs.items():
        setattr(module, attr, value)

    return module


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch, **modules: dict[str, Any]) -> None:
    """Put a minimal fake ``ptycho_torch`` (and ``torch``) on ``sys.modules`` for one test.

    ``modules`` maps a dotted submodule name, with dots spelled ``_DOT_``, to the
    attributes that submodule should carry.
    """
    stand_ins = {
        'ptycho_torch': _fake_module('ptycho_torch', __path__=[]),
        'ptycho_torch.workflows': _fake_module('ptycho_torch.workflows', __path__=[]),
        # _resolve_device asks torch whether a device is present; say no, so the
        # call resolves to CPU without a real runtime.
        'torch': _fake_module('torch', cuda=SimpleNamespace(is_available=lambda: False)),
    }

    for dotted, attrs in modules.items():
        name = dotted.replace('_DOT_', '.')
        stand_ins[name] = _fake_module(name, **attrs)

    for name, module in stand_ins.items():
        monkeypatch.setitem(sys.modules, name, module)


class _FakeExecutionRequest:
    def __init__(self, values: Any, explicit_fields: Any) -> None:
        self.values = dict(values)
        self.explicit_fields = set(explicit_fields)


class _TrainingDoor:
    """Stands in for ``ptycho_torch.train.train``, recording what it was handed."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    def __call__(self, dataset: Path, output_dir: Path, settings: Any, **kwargs: Any) -> Path:
        execution_config = kwargs['execution_config']

        with numpy.load(dataset, allow_pickle=False) as archive:
            dataset_keys = sorted(archive.files)
            num_frames = len(archive['xcoords'])
            object_indexes = (
                sorted({int(i) for i in archive['object_index']})
                if 'object_index' in archive.files
                else None
            )

        self.captured = {
            'dataset_name': str(dataset.name),
            'dataset_parent': str(dataset.parent),
            'dataset_keys': dataset_keys,
            'num_frames': num_frames,
            'object_indexes': object_indexes,
            'settings_is_mapping': isinstance(settings, dict),
            'has_nphotons': 'nphotons' in settings,
            'mode': settings.get('mode'),
            'execution_values': execution_config.values,
            'explicit_fields': sorted(execution_config.explicit_fields),
        }
        return output_dir / 'bundle'


def _train(
    monkeypatch: pytest.MonkeyPatch,
    options: PtychoPINNOptions,
    input_directory: Path,
    output_directory: Path,
) -> tuple[dict[str, Any], list[TrainOutput]]:
    """What training hands the stubbed door, and what it yields, given these options."""
    door = _TrainingDoor()
    _install_fake_backend(
        monkeypatch,
        ptycho_torch_DOT_execution_request={'ExecutionRequest': _FakeExecutionRequest},
        ptycho_torch_DOT_train={'train': door},
    )
    reconstructor = PtychoPINNReconstructor('Unsupervised')
    outputs = list(reconstructor.train(input_directory, output_directory, options=options))
    return door.captured, outputs


def _unsupervised_options(accelerator: Accelerator, num_workers: int) -> PtychoPINNOptions:
    options = default_options()
    options.training.epochs = 3
    options.training.accelerator = accelerator
    options.training.num_dataloader_workers = num_workers
    return options


def _write_training_archive(file_path: Path, num_frames: int) -> None:
    numpy.savez(
        file_path,
        xcoords=numpy.arange(num_frames, dtype=float),
        ycoords=numpy.arange(num_frames, dtype=float),
        diff3d=numpy.ones((num_frames, DETECTOR_PX, DETECTOR_PX)),
        probeGuess=numpy.ones((DETECTOR_PX, DETECTOR_PX), dtype=complex),
        objectGuess=numpy.ones((32, 32), dtype=complex),
    )


@pytest.fixture
def one_archive(tmp_path: Path) -> Path:
    directory = tmp_path / 'input'
    directory.mkdir()
    _write_training_archive(directory / 'train_data.npz', NUM_POSITIONS)
    return directory


def test_train_hands_the_public_door_a_settings_mapping(
    monkeypatch: pytest.MonkeyPatch, one_archive: Path, tmp_path: Path
) -> None:
    """Training goes through the dataset-to-bundle door, not a config object."""
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    captured, _ = _train(
        monkeypatch, _unsupervised_options(Accelerator.CPU, 2), one_archive, tmp_path / 'output'
    )
    # A lone archive is handed over as it is, whatever its name.
    assert captured['dataset_name'] == 'train_data.npz'
    assert captured['dataset_parent'] == str(one_archive)
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


def test_train_names_the_bundle_and_leaves_loading_to_the_caller(
    monkeypatch: pytest.MonkeyPatch, one_archive: Path, tmp_path: Path
) -> None:
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    output_directory = tmp_path / 'output'
    reconstructor = PtychoPINNReconstructor('Unsupervised')
    door = _TrainingDoor()
    _install_fake_backend(
        monkeypatch,
        ptycho_torch_DOT_execution_request={'ExecutionRequest': _FakeExecutionRequest},
        ptycho_torch_DOT_train={'train': door},
    )

    outputs = list(
        reconstructor.train(
            one_archive, output_directory, options=_unsupervised_options(Accelerator.CPU, 0)
        )
    )

    assert [output.trained_model for output in outputs] == [output_directory / 'bundle']
    # The log could not be read, so the epoch count stands in for the progress.
    assert outputs[-1].progress == outputs[-1].progress_goal == 3
    assert not reconstructor.is_model_loaded()


def test_train_under_a_distributed_launcher_claims_one_device_per_local_rank(
    monkeypatch: pytest.MonkeyPatch, one_archive: Path, tmp_path: Path
) -> None:
    """Lightning's TorchElastic environment rejects a device count other than its own."""
    # As a launcher's variables are.
    monkeypatch.setenv('LOCAL_WORLD_SIZE', '2')
    captured, _ = _train(
        monkeypatch, _unsupervised_options(Accelerator.GPU, 0), one_archive, tmp_path / 'output'
    )

    assert captured['execution_values']['devices'] == 2
    assert 'devices' in captured['explicit_fields']


def test_train_merges_several_archives_into_one_object_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv('LOCAL_WORLD_SIZE', raising=False)
    input_directory = tmp_path / 'input'
    input_directory.mkdir()
    _write_training_archive(input_directory / 'train_data_0000.npz', 3)
    _write_training_archive(input_directory / 'train_data_0001.npz', 5)
    (input_directory / 'notes.npz').write_bytes(b'not an archive')
    output_directory = tmp_path / 'output'

    captured, _ = _train(
        monkeypatch, _unsupervised_options(Accelerator.CPU, 2), input_directory, output_directory
    )

    assert captured['num_frames'] == 8
    assert captured['object_indexes'] == [0, 1]
    # One canvas cannot hold two objects, and unsupervised training needs no patches.
    assert 'objectGuess' not in captured['dataset_keys']
    assert 'Y' not in captured['dataset_keys']
    assert Path(captured['dataset_parent']).parent == output_directory
    # The private merge directory is gone once training returns, or fails.
    assert list(output_directory.iterdir()) == []


def test_train_refuses_a_directory_without_archives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_directory = tmp_path / 'input'
    input_directory.mkdir()

    with pytest.raises(FileNotFoundError):
        _train(
            monkeypatch,
            _unsupervised_options(Accelerator.CPU, 2),
            input_directory,
            tmp_path / 'output',
        )


@dataclass
class _FakeInferenceConfig:
    patch_weighting: str = 'probe'
    batch_size: int = 1


class _InferenceDoor:
    """Stands in for ``reconstruct_from_arrays``, recording what it was handed."""

    def __init__(self) -> None:
        self.captured: dict[str, Any] = {}

    def __call__(self, model: Any, arrays: Any, **kwargs: Any) -> Any:
        runtime_params = kwargs['runtime_params']
        self.captured = {
            'array_keys': sorted(arrays),
            'device': kwargs['device'],
            'num_workers': kwargs['num_workers'],
            'patch_weighting': runtime_params.inference_config.patch_weighting,
            'batch_size': runtime_params.inference_config.batch_size,
        }
        return SimpleNamespace(complex_canvas=numpy.ones((1, 32, 32), dtype=complex))


def test_reconstruct_refuses_without_a_loaded_model() -> None:
    reconstructor = PtychoPINNReconstructor('Unsupervised')

    with pytest.raises(RuntimeError, match='no PtychoPINN model has been loaded'):
        list(reconstructor.reconstruct(_make_reconstruct_input(), options=default_options()))


def test_reconstruct_stages_declared_arrays_through_the_embedder_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded_model = SimpleNamespace(
        model_config=SimpleNamespace(mode='Unsupervised'),
        inference_config=_FakeInferenceConfig(),
        data_config=object(),
        training_config=object(),
    )
    door = _InferenceDoor()
    _install_fake_backend(
        monkeypatch,
        ptycho_torch_DOT_inference={
            'ReconstructionRuntimeParams': SimpleNamespace,
            'reconstruct_from_arrays': door,
        },
        ptycho_torch_DOT_workflows_DOT_bundle_io={
            'load_inference_bundle_torch': lambda _dir: (loaded_model, {}),
        },
    )
    options = default_options()
    options.inference.patch_weighting_method = PatchWeighting.UNIFORM
    options.inference.batch_size = 64
    options.training.accelerator = Accelerator.AUTO
    options.training.num_dataloader_workers = 3
    # Named as the backend names it: the loaded bundle's mode is compared against it.
    reconstructor = PtychoPINNReconstructor('Unsupervised')
    reconstructor.load_model(tmp_path)

    outputs = list(reconstructor.reconstruct(_make_reconstruct_input(), options=options))

    assert len(outputs) == 1
    captured = door.captured
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
    numpy.testing.assert_array_equal(
        probe_guess, parameters.product.probes.get_mean_probe().get_array()
    )


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
