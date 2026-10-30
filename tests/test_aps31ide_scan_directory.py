"""Unit tests for locating an APS 31-ID-E scan's ``eiger_4`` directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.plugins.aps31id_lamni.scan_directory import locate_scan_directory


@pytest.mark.parametrize(
    ('block', 'scan_no'),
    [('S01000-01999', 1234), ('S01000-01099', 1001)],
)
def test_a_scan_is_found_whatever_the_block_size(tmp_path: Path, block: str, scan_no: int) -> None:
    scan_dir = tmp_path / block / f'S{scan_no:05d}'
    scan_dir.mkdir(parents=True)
    (tmp_path / 'S00000-00099' / 'S00042').mkdir(parents=True)

    assert locate_scan_directory(tmp_path, scan_no) == scan_dir


def test_a_missing_scan_is_an_error(tmp_path: Path) -> None:
    (tmp_path / 'S01000-01099' / 'S01001').mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        locate_scan_directory(tmp_path, 1002)


def test_a_scan_under_several_blocks_is_an_error(tmp_path: Path) -> None:
    (tmp_path / 'S01000-01099' / 'S01001').mkdir(parents=True)
    (tmp_path / 'S01000-01999' / 'S01001').mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match='several blocks'):
        locate_scan_directory(tmp_path, 1001)
