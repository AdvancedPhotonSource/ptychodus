"""``ptychodus-train`` against a fake library, single-process and as one rank of many.

The library trains in process and writes a stand-in model, so these tests pin down what the
command decides -- which algorithm, which options, whether a launch may go ahead, which rank
writes -- without a backend, a child process or a GPU. Launches are simulated by the
environment variables a launcher sets, which is all the command reads of one.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, Field
import pytest
import yaml

from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import LossValue
from ptychodus.api.reconstruct import (
    NullReconstructor,
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
    TrainOutput,
)
from ptychodus.cli import train

_LAUNCHER_VARIABLES = (
    'RANK',
    'SLURM_PROCID',
    'WORLD_SIZE',
    'SLURM_NTASKS',
    'GROUP_WORLD_SIZE',
    'SLURM_NNODES',
)


class _Training(BaseModel):
    epochs: int = Field(2, ge=1)


class _Options(BaseModel):
    learning_rate: float = Field(1.0e-3, gt=0.0)
    training: _Training = Field(default_factory=_Training)


class _FakeTrainer(TrainableReconstructor):
    def __init__(self, name: str, *, distributed: bool) -> None:
        self._name = name
        self._distributed = distributed
        self._model_path: Path | None = None
        self.fail = False
        self.report_model = True
        self.trained_with: BaseModel | None = None

    @property
    def name(self) -> str:
        return self._name

    def get_default_options(self) -> BaseModel:
        return _Options()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        yield from ()

    def is_model_loaded(self) -> bool:
        return self._model_path is not None

    def get_model_file_filter(self) -> str:
        return 'Fake Model (*.bin)'

    def load_model(self, file_path: Path) -> None:
        self._model_path = file_path

    def get_model_file_extension(self) -> str:
        return '.bin'

    def save_model(self, file_path: Path) -> None:
        assert self._model_path is not None
        shutil.copyfile(self._model_path, file_path)

    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        self.trained_with = options
        yield TrainOutput(training_loss=[LossValue(0, 1.0)], progress=1, progress_goal=2)

        if self.fail:
            raise RuntimeError('diverged')

        weights = output_directory / 'weights.bin'
        weights.write_bytes(b'weights')

        if self.report_model:
            self._model_path = weights

        yield TrainOutput(training_loss=[LossValue(1, 0.5)], progress=2, progress_goal=2)

    def supports_distributed_training(self) -> bool:
        return self._distributed


class _FakeLibrary(ReconstructorLibrary):
    def __init__(self, trainers: Iterable[Reconstructor]) -> None:
        super().__init__('fake', 'fake-distribution')
        self._offered = list(trainers)

    def _build(self) -> Iterable[Reconstructor]:
        return self._offered


@pytest.fixture
def ddp() -> _FakeTrainer:
    return _FakeTrainer('DDP', distributed=True)


@pytest.fixture
def single() -> _FakeTrainer:
    return _FakeTrainer('Single', distributed=False)


@pytest.fixture(autouse=True)
def _registry(monkeypatch: pytest.MonkeyPatch, ddp: _FakeTrainer, single: _FakeTrainer) -> None:
    """Offer only the fake library, and keep ``main()`` off the test process's globals."""
    registry = PluginRegistry()
    registry.reconstructor_libraries.register_plugin(
        _FakeLibrary([ddp, single, NullReconstructor('Solve')]),
        simple_name='fake',
        display_name='Fake',
    )
    registry.reconstructor_libraries.freeze()

    monkeypatch.setattr(PluginRegistry, 'load_plugins', lambda: registry)
    monkeypatch.setattr(train.signal, 'signal', lambda signalnum, handler: None)
    monkeypatch.setattr(train, 'configure_logging', lambda args: None)

    for name in _LAUNCHER_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-train', *map(str, argv)])
    return train.main()


def _usage_error(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, *argv)

    assert exit_info.value.code == ExitCode.USAGE


def _directories(tmp_path: Path) -> tuple[Path, Path]:
    input_directory = tmp_path / 'input'
    input_directory.mkdir()
    return input_directory, tmp_path / 'output'


def test_training_writes_the_model_under_the_standard_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ddp: _FakeTrainer
) -> None:
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'ddp',
    )  # fmt: skip

    assert code == ExitCode.SUCCESS
    assert (output_directory / 'model.bin').read_bytes() == b'weights'
    assert ddp.trained_with == _Options()


def test_printed_options_load_back_and_select_their_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    single: _FakeTrainer,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _main(monkeypatch, '--print-options', 'Fake/single')

    assert exit_info.value.code == 0
    document = yaml.safe_load(capsys.readouterr().out)
    assert (document['library'], document['algorithm']) == ('fake', 'Single')

    document['options']['training']['epochs'] = 5
    options_file = tmp_path / 'options.yaml'
    options_file.write_text(yaml.safe_dump(document))
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--options', options_file
    )

    assert code == ExitCode.SUCCESS
    assert single.trained_with == _Options(training=_Training(epochs=5))


def test_an_ambiguous_library_needs_an_algorithm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake')


def test_an_algorithm_that_cannot_train_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'Solve',
    )  # fmt: skip


def test_flags_that_disagree_with_the_options_file_are_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    options_file = tmp_path / 'options.yaml'
    options_file.write_text('library: fake\nalgorithm: Single\n')
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--options', options_file,
        '--algorithm', 'DDP',
    )  # fmt: skip


def test_an_undeclared_option_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    options_file = tmp_path / 'options.yaml'
    options_file.write_text(
        'library: fake\nalgorithm: Single\noptions:\n  training:\n    epoch: 3\n'
    )
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--options', options_file
    )


def test_the_output_directory_must_differ_from_the_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_directory, _ = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', input_directory, '--library', 'fake',
        '--algorithm', 'DDP',
    )  # fmt: skip


@pytest.mark.parametrize('variable', ['WORLD_SIZE', 'SLURM_NTASKS'])
def test_a_backend_that_trains_alone_is_refused_across_processes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, single: _FakeTrainer, variable: str
) -> None:
    monkeypatch.setenv(variable, '2')
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'Single',
    )  # fmt: skip

    assert single.trained_with is None
    assert not output_directory.exists()


@pytest.mark.parametrize('variable', ['GROUP_WORLD_SIZE', 'SLURM_NNODES'])
def test_a_launch_across_nodes_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str
) -> None:
    monkeypatch.setenv('WORLD_SIZE', '4')
    monkeypatch.setenv(variable, '2')
    input_directory, output_directory = _directories(tmp_path)

    _usage_error(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'DDP',
    )  # fmt: skip


@pytest.mark.parametrize(('rank', 'writes_model'), [('0', True), ('1', False)])
def test_only_the_main_process_saves_the_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ddp: _FakeTrainer,
    rank: str,
    writes_model: bool,
) -> None:
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('RANK', rank)
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'DDP',
    )  # fmt: skip

    assert code == ExitCode.SUCCESS
    assert ddp.trained_with is not None
    assert (output_directory / 'model.bin').exists() is writes_model


def test_a_failed_training_reports_failure_and_saves_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ddp: _FakeTrainer
) -> None:
    ddp.fail = True
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'DDP',
    )  # fmt: skip

    assert code == ExitCode.FAILURE
    assert not (output_directory / 'model.bin').exists()


def test_a_training_that_reports_no_model_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ddp: _FakeTrainer
) -> None:
    ddp.report_model = False
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'DDP',
    )  # fmt: skip

    assert code == ExitCode.FAILURE


def test_a_dry_run_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ddp: _FakeTrainer
) -> None:
    input_directory, output_directory = _directories(tmp_path)

    code = _main(
        monkeypatch, '-i', input_directory, '-o', output_directory, '--library', 'fake',
        '--algorithm', 'DDP', '--dry-run',
    )  # fmt: skip

    assert code == ExitCode.SUCCESS
    assert ddp.trained_with is None
    assert not output_directory.exists()
