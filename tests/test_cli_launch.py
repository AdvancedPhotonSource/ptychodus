"""A distributed launcher's report of this process, read from the environment.

Each reader takes the torch name before the SLURM one and falls back to the
single-process value, skipping any variable that does not parse rather than failing on it.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ptychodus.cli._launch import (
    environment_node_count,
    environment_rank,
    environment_world_size,
)

_READERS: list[tuple[Callable[[], int], str, str, int]] = [
    (environment_rank, 'RANK', 'SLURM_PROCID', 0),
    (environment_world_size, 'WORLD_SIZE', 'SLURM_NTASKS', 1),
    (environment_node_count, 'GROUP_WORLD_SIZE', 'SLURM_NNODES', 1),
]


@pytest.fixture
def no_launcher(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for _, torch_name, slurm_name, _ in _READERS:
        monkeypatch.delenv(torch_name, raising=False)
        monkeypatch.delenv(slurm_name, raising=False)

    return monkeypatch


@pytest.mark.parametrize(('reader', 'torch_name', 'slurm_name', 'default'), _READERS)
def test_without_a_launcher_reads_as_a_single_process(
    no_launcher: pytest.MonkeyPatch,
    reader: Callable[[], int],
    torch_name: str,
    slurm_name: str,
    default: int,
) -> None:
    assert reader() == default


@pytest.mark.parametrize(('reader', 'torch_name', 'slurm_name', 'default'), _READERS)
def test_the_torch_name_wins_over_the_slurm_one(
    no_launcher: pytest.MonkeyPatch,
    reader: Callable[[], int],
    torch_name: str,
    slurm_name: str,
    default: int,
) -> None:
    no_launcher.setenv(torch_name, '3')
    no_launcher.setenv(slurm_name, '5')

    assert reader() == 3


@pytest.mark.parametrize(('reader', 'torch_name', 'slurm_name', 'default'), _READERS)
def test_slurm_alone_is_read(
    no_launcher: pytest.MonkeyPatch,
    reader: Callable[[], int],
    torch_name: str,
    slurm_name: str,
    default: int,
) -> None:
    no_launcher.setenv(slurm_name, '5')

    assert reader() == 5


@pytest.mark.parametrize(('reader', 'torch_name', 'slurm_name', 'default'), _READERS)
@pytest.mark.parametrize('value', ['', 'oops'])
def test_an_unreadable_variable_is_skipped(
    no_launcher: pytest.MonkeyPatch,
    reader: Callable[[], int],
    torch_name: str,
    slurm_name: str,
    default: int,
    value: str,
) -> None:
    no_launcher.setenv(torch_name, value)

    assert reader() == default

    no_launcher.setenv(slurm_name, '5')

    assert reader() == 5
