"""Unit tests for the APS 31-ID-E ``tomography_scannumbers.txt`` reader.

The fixture carries rows copied from a real 2026-1 experiment, trimmed to a handful and
with two deliberately malformed rows appended, since the table is appended to live during
an experiment and a reader of it can catch a partially written line.
"""

from pathlib import Path
import logging

import pytest

from ptychodus.plugins.aps31id_lamni._scan_table import (
    APS31IDEScanRecord,
    read_aps31ide_scan_table,
)

DATA_DIR = Path(__file__).parent / 'data' / 'lamni'


@pytest.fixture
def records() -> list[APS31IDEScanRecord]:
    return read_aps31ide_scan_table(DATA_DIR / 'tomography_scannumbers.txt')


def test_fields_are_typed(records: list[APS31IDEScanRecord]) -> None:
    """The angle must arrive as a number: it becomes a product's tomography_angle_deg."""
    first = records[0]

    assert first == APS31IDEScanRecord(
        scan_no=188,
        golden_angle_deg=0.0,
        encoder_angle_deg=-90.0461,
        measurement_id=1,
        subtomo_no=1,
        detector_position=1,
        label='BOREAS_3d_50um',
    )
    assert isinstance(first.encoder_angle_deg, float)
    assert isinstance(first.scan_no, int)


def test_rows_are_returned_in_file_order(records: list[APS31IDEScanRecord]) -> None:
    """Order is acquisition order, which callers rely on; it is not sorted or grouped."""
    assert [record.scan_no for record in records] == [188, 189, 190, 368, 369, 5, 6]


def test_every_label_is_returned(records: list[APS31IDEScanRecord]) -> None:
    """One table holds every tomogram of an experiment; selecting one is the caller's job."""
    assert {record.label for record in records} == {'BOREAS_3d_50um', 'BOREAS_CoR'}


def test_comment_rows_are_skipped(records: list[APS31IDEScanRecord]) -> None:
    assert all(not record.label.startswith('#') for record in records)
    assert len(records) == 7


def test_repeated_projections_are_all_returned(records: list[APS31IDEScanRecord]) -> None:
    """A re-run appends a row under a new scan number; the table does not say which wins."""
    repeats = [r for r in records if (r.subtomo_no, r.measurement_id) == (2, 1)]

    assert [record.scan_no for record in repeats] == [368, 369]


def test_short_row_is_skipped_with_a_warning(
    records: list[APS31IDEScanRecord], caplog: pytest.LogCaptureFixture
) -> None:
    """Scan 191's row has six columns, so it cannot be assigned to fields."""
    assert all(record.scan_no != 191 for record in records)

    with caplog.at_level(logging.WARNING):
        read_aps31ide_scan_table(DATA_DIR / 'tomography_scannumbers.txt')

    assert 'Unexpected row' in caplog.text


def test_unparsable_field_is_skipped_with_a_warning(
    records: list[APS31IDEScanRecord], caplog: pytest.LogCaptureFixture
) -> None:
    """Scan 192's measurement_id is not a number."""
    assert all(record.scan_no != 192 for record in records)

    with caplog.at_level(logging.WARNING):
        read_aps31ide_scan_table(DATA_DIR / 'tomography_scannumbers.txt')

    assert 'Failed to parse row' in caplog.text


def test_empty_table_yields_no_records(tmp_path: Path) -> None:
    file_path = tmp_path / 'empty.txt'
    file_path.write_text('')

    assert read_aps31ide_scan_table(file_path) == []
