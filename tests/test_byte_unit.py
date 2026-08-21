"""Unit tests for ByteUnit and format_bytes."""

from __future__ import annotations

import pytest

from ptychodus.api.constants import ByteUnit, format_bytes


@pytest.mark.parametrize(
    'nbytes, expected',
    [
        (0, ByteUnit.B),
        (1, ByteUnit.B),
        (999, ByteUnit.B),
        (1000, ByteUnit.KB),
        (999_999, ByteUnit.KB),
        (1_000_000, ByteUnit.MB),
        (4_100_000_000, ByteUnit.GB),
        (7 * 1000**4, ByteUnit.TB),
        (3 * 1000**5, ByteUnit.PB),
    ],
)
def test_from_bytes_picks_the_largest_fitting_unit(nbytes: int, expected: ByteUnit) -> None:
    assert ByteUnit.from_bytes(nbytes) is expected


def test_from_bytes_clamps_above_the_largest_unit() -> None:
    assert ByteUnit.from_bytes(10**30) is ByteUnit.PB


def test_from_bytes_clamps_below_zero() -> None:
    assert ByteUnit.from_bytes(-1) is ByteUnit.B


def test_convert_expresses_a_count_in_the_chosen_unit() -> None:
    assert ByteUnit.MB.convert(4_096_000_000) == pytest.approx(4096.0)
    assert ByteUnit.GB.convert(4_096_000_000) == pytest.approx(4.096)
    assert ByteUnit.B.convert(512) == pytest.approx(512.0)


def test_format_in_a_fixed_unit() -> None:
    assert ByteUnit.MB.format(12_340_000) == '12.34 MB'
    assert ByteUnit.GB.format(12_340_000) == '0.01 GB'


def test_bytes_render_as_a_bare_integer() -> None:
    assert ByteUnit.B.format(0) == '0 B'
    assert ByteUnit.B.format(512) == '512 B'


@pytest.mark.parametrize(
    'nbytes, expected',
    [
        (0, '0 B'),
        (512, '512 B'),
        (999, '999 B'),
        (1000, '1.00 kB'),
        (999_999, '1000.00 kB'),
        (1_000_000, '1.00 MB'),
        (12_340_000, '12.34 MB'),
        (4_100_000_000, '4.10 GB'),
    ],
)
def test_format_bytes_end_to_end(nbytes: int, expected: str) -> None:
    assert format_bytes(nbytes) == expected


def test_bytes_per_unit_ladder_is_decimal() -> None:
    assert ByteUnit.B.bytes_per_unit == 1
    assert ByteUnit.KB.bytes_per_unit == 1000
    assert ByteUnit.MB.bytes_per_unit == 1000**2
    assert ByteUnit.GB.bytes_per_unit == 1000**3
