"""The action log every reconstruct, train and enhance command writes into its output.

A log is named for the action and appended to rather than replaced, so a directory a second
run writes into keeps the first run's records; each process of a multi-process launch keeps
its own file. The descriptor-capturing mode is not exercised here: it repoints the test
process's own standard streams, which a test cannot give back.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ptychodus.api.io import StandardFileLayout
from ptychodus.cli import open_action_log


def test_the_log_is_named_for_the_action_and_records_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr('sys.argv', ['ptychodus-train', '-i', 'in dir', '-o', 'out'])
    output_directory = tmp_path / 'created'

    log_file = open_action_log(output_directory, StandardFileLayout.TRAIN_LOG)
    logging.getLogger('test_cli_logging').warning('trained')

    assert log_file == output_directory / 'train.log'
    first, second = log_file.read_text().splitlines()
    assert first.startswith('# ') and first.endswith("ptychodus-train -i 'in dir' -o out")
    assert second.endswith('test_cli_logging: trained')


def test_a_second_run_appends(tmp_path: Path) -> None:
    log_file = StandardFileLayout.RECONSTRUCT_LOG.path(tmp_path)
    log_file.write_text('earlier run\n')

    open_action_log(tmp_path, StandardFileLayout.RECONSTRUCT_LOG)

    assert log_file.read_text().startswith('earlier run\n')


def test_a_secondary_rank_logs_to_its_own_file(tmp_path: Path) -> None:
    # Every rank logs, and one shared name would have them interleaving one file.
    log_file = open_action_log(tmp_path, StandardFileLayout.RECONSTRUCT_LOG, rank=12)

    assert log_file.name == 'reconstruct.rank12.log'
    assert not StandardFileLayout.RECONSTRUCT_LOG.path(tmp_path).exists()
