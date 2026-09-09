"""Unit tests for the APS 31-ID-E LamNI probe-position reader.

The fixtures in ``tests/data/lamni`` carry headers copied verbatim from real
beamline files, with the bodies trimmed to a handful of rows. Coordinate values
in the softGlueZynq fixtures are the synthetic placeholders the instrument team
ships in its own example files.
"""

from pathlib import Path

import pytest

from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.aps31id_lamni_position_file import LamNIPositionFileReader

DATA_DIR = Path(__file__).parent / 'data' / 'lamni'

MICROMETER_M = 1e-6


@pytest.fixture
def reader() -> LamNIPositionFileReader:
    return LamNIPositionFileReader()


def test_orchestra_layout(reader: LamNIPositionFileReader) -> None:
    """Orchestra indexes by DataPoint, which already runs from zero."""
    positions = reader.read(DATA_DIR / 'orchestra.dat')

    assert len(positions) == 8
    assert [point.index for point in positions] == list(range(8))

    # Both axes are negated and converted from micrometres.
    assert positions[0].x_m == pytest.approx(-19.50612 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(+9.67084 * MICROMETER_M)
    assert positions[7].x_m == pytest.approx(-15.96804 * MICROMETER_M)
    assert positions[7].y_m == pytest.approx(+10.07311 * MICROMETER_M)

    assert positions.get_probe_photon_counts() is None


def test_soft_glue_zynq_processed_layout(reader: LamNIPositionFileReader) -> None:
    """The processed layout's 1-based Detector_Count is shifted to a 0-based index."""
    positions = reader.read(DATA_DIR / 'softglue_processed.dat')

    assert len(positions) == 5
    assert [point.index for point in positions] == list(range(5))

    assert positions[0].x_m == pytest.approx(+0.1450575 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(-0.1450575 * MICROMETER_M)


def test_soft_glue_zynq_raw_layout_keeps_duplicate_indexes(
    reader: LamNIPositionFileReader,
) -> None:
    """Raw rows are oversampled: several share one trigger, averaged downstream."""
    positions = reader.read(DATA_DIR / 'softglue_raw.dat')

    assert len(positions) == 8
    assert [point.index for point in positions] == [0, 0, 0, 0, 1, 1, 1, 1]

    # x_st_fzp, not Average_x_st_fzp, and still negated.
    assert positions[0].x_m == pytest.approx(-0.12345 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(+0.12345 * MICROMETER_M)


def test_index_origin_agrees_across_layouts(reader: LamNIPositionFileReader) -> None:
    """Regression guard for the off-by-one between the two DAQ paths.

    Orchestra numbers the first scan point ``DataPoint = 0`` while softGlueZynq
    numbers the same point ``Detector_Count = 1``. Both must emit index 0, so
    that either recording of a scan joins to the same pattern.
    """
    orchestra = reader.read(DATA_DIR / 'orchestra.dat')
    raw = reader.read(DATA_DIR / 'softglue_raw.dat')
    processed = reader.read(DATA_DIR / 'softglue_processed.dat')

    assert orchestra[0].index == 0
    assert raw[0].index == 0
    assert processed[0].index == 0


def test_comment_rows_are_skipped(reader: LamNIPositionFileReader, tmp_path: Path) -> None:
    lines = (DATA_DIR / 'softglue_processed.dat').read_text().splitlines()
    lines.insert(3, '# a comment interrupting the data')
    file_path = tmp_path / 'commented.dat'
    file_path.write_text('\n'.join(lines) + '\n')

    positions = reader.read(file_path)

    assert len(positions) == 5
    assert [point.index for point in positions] == list(range(5))


def test_unknown_header_names_every_known_layout(
    reader: LamNIPositionFileReader, tmp_path: Path
) -> None:
    file_path = tmp_path / 'unknown.dat'
    file_path.write_text('Scan 00001, lsamrot_encoder 0.0\nAlpha Beta Gamma\n0 1 2\n')

    with pytest.raises(ProbePositionParseError) as exc_info:
        reader.read(file_path)

    message = str(exc_info.value)
    assert 'Alpha Beta Gamma' in message
    assert 'Orchestra' in message
    assert 'softGlueZynq raw' in message
    assert 'softGlueZynq processed' in message


def test_short_row_is_rejected(reader: LamNIPositionFileReader, tmp_path: Path) -> None:
    lines = (DATA_DIR / 'softglue_processed.dat').read_text().splitlines()
    lines[3] = '2 -0.207402 0.199276675946'
    file_path = tmp_path / 'short_row.dat'
    file_path.write_text('\n'.join(lines) + '\n')

    with pytest.raises(ProbePositionParseError, match='Bad number of columns'):
        reader.read(file_path)
