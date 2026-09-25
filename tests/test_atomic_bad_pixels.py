"""Tests for the Atomic JSON bad-pixel list.

The file names masked pixels as ``[column, row]`` and carries no detector dimensions, so
every test here is about the two things the reader has to supply from outside the file:
the axis order and the extent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy
import pytest

from ptychodus.plugins.bad_pixels import AtomicBadPixelsFileReader

_HEIGHT = AtomicBadPixelsFileReader.DETECTOR_EXTENT.height_px
_WIDTH = AtomicBadPixelsFileReader.DETECTOR_EXTENT.width_px


def _write(path: Path, contents: Any) -> Path:
    path.write_text(json.dumps(contents))
    return path


def _write_pixels(path: Path, pixel_list: list[list[int]]) -> Path:
    key = AtomicBadPixelsFileReader.BAD_PIXELS_KEY
    entry_list = [{'Pixel': pixel, 'Set': 0} for pixel in pixel_list]
    return _write(path, {key: entry_list})


def test_pixels_are_column_row(tmp_path: Path) -> None:
    """Transposing the pair is silent on a square detector and wrong on this one."""
    file_path = _write_pixels(tmp_path / 'bad.json', [[0, 0], [7, 3], [_WIDTH - 1, _HEIGHT - 1]])

    bad_pixels = AtomicBadPixelsFileReader().read(file_path)

    assert bad_pixels.shape == (_HEIGHT, _WIDTH)
    assert bad_pixels.dtype == numpy.bool_
    assert bad_pixels[0, 0]
    assert bad_pixels[3, 7]
    assert bad_pixels[_HEIGHT - 1, _WIDTH - 1]
    assert not bad_pixels[7, 3]
    assert bad_pixels.sum() == 3


def test_empty_list_masks_nothing(tmp_path: Path) -> None:
    bad_pixels = AtomicBadPixelsFileReader().read(_write_pixels(tmp_path / 'bad.json', []))

    assert bad_pixels.shape == (_HEIGHT, _WIDTH)
    assert not bad_pixels.any()


def test_repeated_pixel_is_idempotent(tmp_path: Path) -> None:
    file_path = _write_pixels(tmp_path / 'bad.json', [[4, 5], [4, 5]])

    assert AtomicBadPixelsFileReader().read(file_path).sum() == 1


def test_set_value_is_not_interpreted(tmp_path: Path) -> None:
    """A pixel is bad by being listed; `Set` selects nothing."""
    key = AtomicBadPixelsFileReader.BAD_PIXELS_KEY
    file_path = _write(
        tmp_path / 'bad.json',
        {key: [{'Pixel': [1, 2], 'Set': 0}, {'Pixel': [3, 4], 'Set': 7}]},
    )

    assert AtomicBadPixelsFileReader().read(file_path).sum() == 2


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'bad.json', {'Pixels': []})

    with pytest.raises(ValueError, match='Bad pixels'):
        AtomicBadPixelsFileReader().read(file_path)


def test_malformed_entry_is_rejected(tmp_path: Path) -> None:
    key = AtomicBadPixelsFileReader.BAD_PIXELS_KEY
    file_path = _write(tmp_path / 'bad.json', {key: [{'Pixel': [1, 2]}, {'Set': 0}]})

    with pytest.raises(ValueError, match='Entry 1'):
        AtomicBadPixelsFileReader().read(file_path)


@pytest.mark.parametrize(
    'pixel, axis',
    [
        ([_WIDTH, 0], 'column'),
        ([0, _HEIGHT], 'row'),
        ([-1, 0], 'column'),
        ([0, -1], 'row'),
    ],
)
def test_out_of_range_pixel_is_rejected(tmp_path: Path, pixel: list[int], axis: str) -> None:
    """A coordinate off the detector means the wrong detector, not a bigger mask.

    Sizing the array to fit would hand back a mask whose shape depends on which pixels
    happen to be listed, and a too-small one reaches the shape check in
    SimpleDiffractionDataset far from its cause.
    """
    file_path = _write_pixels(tmp_path / 'bad.json', [pixel])

    with pytest.raises(ValueError, match=axis):
        AtomicBadPixelsFileReader().read(file_path)
