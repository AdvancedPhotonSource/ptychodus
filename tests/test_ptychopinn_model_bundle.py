"""Saving a trained PtychoPINN model, and loading back what was saved.

A bundle is its ``wts.h5.zip`` alone, so a save wraps that one archive and leaves
whatever else training wrote beside it. Nothing here imports ptycho_torch: saving and
unpacking are pure zipfile work done parent-side.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ptychodus.plugins.ptychopinn.reconstructor import (
    MODEL_FILE_NAME,
    _extract_bundle_dir,
    _save_model_bundle,
)

_WEIGHTS = b'weights and manifest'


def _training_output(directory: Path) -> Path:
    """A directory shaped like PtychoPINN's training output: the bundle and its leftovers."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MODEL_FILE_NAME).write_bytes(_WEIGHTS)
    (directory / 'metrics.csv').write_text('epoch,train_loss\n0,1.0\n')
    (directory / 'checkpoints').mkdir()
    (directory / 'checkpoints' / 'last.ckpt').write_bytes(b'checkpoint')
    return directory


@pytest.mark.parametrize('recorded', ['directory', 'archive'])
def test_a_save_holds_the_bundle_archive_and_nothing_else(tmp_path: Path, recorded: str) -> None:
    output = _training_output(tmp_path / 'output')
    loaded_from = output if recorded == 'directory' else output / MODEL_FILE_NAME
    dest = tmp_path / 'model.zip'

    _save_model_bundle(loaded_from, dest)

    with zipfile.ZipFile(dest) as outer:
        assert outer.namelist() == [MODEL_FILE_NAME]
        assert outer.read(MODEL_FILE_NAME) == _WEIGHTS


def test_a_save_inside_the_bundle_directory_does_not_archive_itself(tmp_path: Path) -> None:
    output = _training_output(tmp_path / 'output')
    dest = output / 'model.zip'

    _save_model_bundle(output / MODEL_FILE_NAME, dest)

    with zipfile.ZipFile(dest) as outer:
        assert outer.namelist() == [MODEL_FILE_NAME]

    assert sorted(path.name for path in output.iterdir()) == [
        'checkpoints',
        'metrics.csv',
        'model.zip',
        MODEL_FILE_NAME,
    ]


def test_a_save_is_written_exactly_where_named(tmp_path: Path) -> None:
    output = _training_output(tmp_path / 'output')
    dest = tmp_path / 'trained.bundle'

    _save_model_bundle(output, dest)

    assert zipfile.is_zipfile(dest)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['output', 'trained.bundle']


def test_a_saved_model_loads_back(tmp_path: Path) -> None:
    dest = tmp_path / 'model.zip'
    _save_model_bundle(_training_output(tmp_path / 'output'), dest)

    bundle_dir = _extract_bundle_dir(dest)

    assert (bundle_dir / MODEL_FILE_NAME).read_bytes() == _WEIGHTS


def test_a_bundle_directory_without_its_archive_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()
    dest = tmp_path / 'model.zip'

    with pytest.raises(RuntimeError, match='is not a file'):
        _save_model_bundle(empty, dest)

    assert not dest.exists()
