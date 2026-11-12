"""Tests for finding PtychoPINN training archives in a directory and merging them into a bank."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy
import pytest

from ptychodus.plugins.ptychopinn.training_file import (
    find_ptychopinn_training_archives,
    merge_ptychopinn_training_archives,
)
from ptychodus.api.typing import ComplexArrayType


def _write_archive(
    file_path: Path,
    *,
    num_frames: int = 5,
    patch_size: int = 8,
    probe_scale: float = 1.0,
    seed: int = 0,
    contract: bool = True,
    **extra: Any,
) -> Path:
    rng = numpy.random.default_rng(seed)
    contents: dict[str, Any] = {
        'xcoords': rng.uniform(10.0, 20.0, num_frames),
        'ycoords': rng.uniform(10.0, 20.0, num_frames),
        'diff3d': rng.poisson(5.0, (num_frames, patch_size, patch_size)).astype(float),
        'probeGuess': probe_scale * numpy.ones((patch_size, patch_size), dtype=complex),
        'objectGuess': numpy.ones((32, 32), dtype=complex),
        'bad_pixels': numpy.zeros((patch_size, patch_size), dtype=bool),
        'scan_index': numpy.full(num_frames, 3, dtype=int),
    }
    contents['xcoords_start'] = contents['xcoords']
    contents['ycoords_start'] = contents['ycoords']

    if contract:
        contents['scale_contract_version'] = 'ci_intensity_v2'
        contents['measurement_domain'] = 'count_intensity'

    contents.update(extra)
    numpy.savez(file_path, **contents)
    return file_path


class TestFindArchives:
    def test_finds_archives_sorted_and_skips_others(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        second = _write_archive(tmp_path / 'b.npz')
        first = _write_archive(tmp_path / 'a.npz')
        numpy.savez(tmp_path / 'unrelated.npz', values=numpy.arange(3))
        (tmp_path / 'broken.npz').write_bytes(b'not an archive')
        (tmp_path / 'notes.txt').write_text('ignored')
        nested = tmp_path / 'nested'
        nested.mkdir()
        _write_archive(nested / 'c.npz')

        with caplog.at_level(logging.WARNING):
            archives = find_ptychopinn_training_archives(tmp_path)

        assert archives == [first, second]
        assert 'unrelated.npz' in caplog.text
        assert 'broken.npz' in caplog.text

    def test_raises_without_archives(self, tmp_path: Path) -> None:
        numpy.savez(tmp_path / 'unrelated.npz', values=numpy.arange(3))

        with pytest.raises(FileNotFoundError):
            find_ptychopinn_training_archives(tmp_path)


class TestMergeArchives:
    def test_concatenates_frames_as_an_object_bank(self, tmp_path: Path) -> None:
        paths = [
            _write_archive(tmp_path / 'a.npz', num_frames=3, seed=1),
            _write_archive(tmp_path / 'b.npz', num_frames=4, seed=2),
        ]

        merged = merge_ptychopinn_training_archives(paths, tmp_path / 'bank')

        assert merged == tmp_path / 'bank.npz'

        with numpy.load(merged, allow_pickle=False) as bank:
            assert bank['diff3d'].shape == (7, 8, 8)
            numpy.testing.assert_array_equal(bank['object_index'], [0, 0, 0, 1, 1, 1, 1])
            numpy.testing.assert_array_equal(bank['scan_index'], numpy.full(7, 3))

            for key in ('diff3d', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start'):
                with numpy.load(paths[0]) as a, numpy.load(paths[1]) as b:
                    expected = numpy.concatenate([a[key], b[key]])
                numpy.testing.assert_array_equal(bank[key], expected)

            assert 'objectGuess' not in bank.files
            assert 'bad_pixels' not in bank.files
            assert 'Y' not in bank.files
            assert str(bank['scale_contract_version']) == 'ci_intensity_v2'
            assert str(bank['measurement_domain']) == 'count_intensity'
            numpy.testing.assert_array_equal(bank['probeGuess'], numpy.ones((8, 8)))

    def test_defaults_optional_frame_fields(self, tmp_path: Path) -> None:
        path = tmp_path / 'minimal.npz'
        rng = numpy.random.default_rng(3)
        numpy.savez(
            path,
            xcoords=rng.uniform(size=4),
            ycoords=rng.uniform(size=4),
            diff3d=numpy.ones((4, 8, 8)),
            probeGuess=numpy.ones((8, 8), dtype=complex),
        )

        merged = merge_ptychopinn_training_archives([path, path], tmp_path / 'bank')

        with numpy.load(merged) as bank:
            numpy.testing.assert_array_equal(bank['xcoords_start'], bank['xcoords'])
            numpy.testing.assert_array_equal(bank['scan_index'], numpy.zeros(8))
            assert 'scale_contract_version' not in bank.files

    def test_stores_ground_truth_patches(self, tmp_path: Path) -> None:
        paths = [
            _write_archive(tmp_path / 'a.npz', num_frames=3),
            _write_archive(tmp_path / 'b.npz', num_frames=2),
        ]
        values = {paths[0]: 1.0 + 1.0j, paths[1]: 2.0 - 1.0j}

        def truth(file_path: Path) -> ComplexArrayType:
            with numpy.load(file_path) as archive:
                return numpy.full(archive['diff3d'].shape, values[file_path], dtype=complex)

        merged = merge_ptychopinn_training_archives(paths, tmp_path / 'bank', truth_patches=truth)

        with numpy.load(merged) as bank:
            assert bank['Y'].shape == (5, 8, 8)
            numpy.testing.assert_array_equal(bank['Y'][:3], 1.0 + 1.0j)
            numpy.testing.assert_array_equal(bank['Y'][3:], 2.0 - 1.0j)

    def test_rejects_mismatched_ground_truth(self, tmp_path: Path) -> None:
        paths = [_write_archive(tmp_path / 'a.npz'), _write_archive(tmp_path / 'b.npz')]

        with pytest.raises(ValueError, match='Ground truth'):
            merge_ptychopinn_training_archives(
                paths, tmp_path / 'bank', truth_patches=lambda _: numpy.ones((1, 8, 8), complex)
            )

    @pytest.mark.parametrize(
        ('second', 'message'),
        [
            ({'probe_scale': 2.0}, 'different probes'),
            ({'patch_size': 16}, 'patterns'),
            ({'contract': False}, 'scale contract'),
            ({'measurement_domain': 'normalized_amplitude'}, 'scale contract'),
        ],
    )
    def test_refuses_archives_that_disagree(
        self, tmp_path: Path, second: dict[str, Any], message: str
    ) -> None:
        paths = [_write_archive(tmp_path / 'a.npz'), _write_archive(tmp_path / 'b.npz', **second)]

        with pytest.raises(ValueError, match=message):
            merge_ptychopinn_training_archives(paths, tmp_path / 'bank')

        assert not (tmp_path / 'bank.npz').exists()

    def test_undeclared_archives_merge_with_each_other(self, tmp_path: Path) -> None:
        paths = [
            _write_archive(tmp_path / 'a.npz', contract=False),
            _write_archive(tmp_path / 'b.npz', contract=False),
        ]

        merged = merge_ptychopinn_training_archives(paths, tmp_path / 'bank')

        with numpy.load(merged) as bank:
            assert 'measurement_domain' not in bank.files
