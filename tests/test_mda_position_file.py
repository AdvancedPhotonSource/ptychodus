"""Unit tests for the EPICS MDA probe-position readers.

MDA is a binary XDR container, and ``tests/data`` is otherwise text-only, so the
fixtures here are synthesized in ``tmp_path`` by ``_write_mda``. That writer emits
XDR with :mod:`struct` directly rather than through ``xdrlib``, which keeps it an
independent oracle for the reader instead of a round-trip through the same code.

The layout it produces mirrors what real 19-ID-E files carry: a flat rank-1 scan
for :class:`MDAFlatScanPositionFileReader`, and a rank-2 scan whose outer readback
holds y while each lower scan holds x for :class:`MDAPositionFileReader`.

``xdrlib`` was removed from the standard library in Python 3.13, so the plugin
falls back to a vendored copy. On 3.11 and 3.12 that fallback never runs, which
would leave the code production depends on untested here; ``test_vendored_xdrlib_*``
forces it on every interpreter.
"""

from __future__ import annotations

from pathlib import Path
import json
import pkgutil
import struct
import subprocess
import sys

import pytest

import ptychodus.plugins
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.mda import _xdrlib, mda_position_file
from ptychodus.plugins.mda.mda_position_file import (
    EpicsType,
    MDADetectorChannelPositionFileReader,
    MDAFile,
    MDAFlatScanPositionFileReader,
    MDAPositionFileReader,
)

MICROMETER_M = 1e-6
MILLIMETER_M = 1e-3

MDA_READER_NAMES = frozenset(
    {'MDA', 'APS_2IDD', 'APS_2IDE', 'APS_Atomic', 'APS_BNP', 'APS_ISN_MDA', 'CNM_APS_HXN'}
)


# --- XDR primitives, written without xdrlib so the fixture is an independent oracle


def _int(value: int) -> bytes:
    return struct.pack('>i', value)


def _float(value: float) -> bytes:
    return struct.pack('>f', value)


def _double(value: float) -> bytes:
    return struct.pack('>d', value)


def _string(value: str) -> bytes:
    """An MDA counted string: the length, then XDR's own length-prefixed string.

    The reader spends an int on the length and then hands the remainder to
    ``Unpacker.unpack_string``, which reads its own length prefix -- so the count
    appears twice. A zero-length string is just the single zero int.
    """
    raw = value.encode()
    length = len(raw)

    if length == 0:
        return _int(0)

    return _int(length) + _int(length) + raw + b'\0' * (-length % 4)


def _positioner(number: int, name: str) -> bytes:
    return (
        _int(number)
        + _string(name)
        + _string(f'{name} description')
        + _string('LINEAR')
        + _string('um')
        + _string(f'{name}.RBV')
        + _string(f'{name} readback')
        + _string('um')
    )


def _detector(number: int, name: str, description: str | None = None) -> bytes:
    return (
        _int(number)
        + _string(name)
        + _string(f'{name} description' if description is None else description)
        + _string('counts')
    )


def _trigger(number: int, name: str, command: float) -> bytes:
    return _int(number) + _string(name) + _float(command)


def _scan_info(
    readbacks: list[list[float]],
    detectors: list[list[float]],
    detector_descriptions: list[str] | None = None,
) -> bytes:
    descriptions = detector_descriptions or [None] * len(detectors)  # type: ignore[list-item]
    return (
        _string('scan1')
        + _string('Jun 19, 2025 13:45:00.000000')
        + _int(len(readbacks))
        + _int(len(detectors))
        + _int(1)
        + b''.join(_positioner(i, f'positioner{i}') for i in range(len(readbacks)))
        + b''.join(_detector(i, f'detector{i}', descriptions[i]) for i in range(len(detectors)))
        + _trigger(0, 'trigger0', 1.0)
    )


def _scan_data(readbacks: list[list[float]], detectors: list[list[float]]) -> bytes:
    doubles = b''.join(_double(v) for row in readbacks for v in row)
    floats = b''.join(_float(v) for row in detectors for v in row)
    return doubles + floats


def _scan(
    readbacks: list[list[float]],
    detectors: list[list[float]],
    *,
    rank: int = 1,
    lower_offsets: list[int] | None = None,
    current_point: int | None = None,
    detector_descriptions: list[str] | None = None,
) -> bytes:
    """Rows are always written full width; current_point says how many are real.

    That is what the instrument does -- an aborted scan leaves the preallocated
    tail zeroed rather than shortening the file.
    """
    npts = len(readbacks[0])
    cpt = npts if current_point is None else current_point
    offsets = lower_offsets or []
    header = _int(rank) + _int(npts) + _int(cpt) + b''.join(_int(o) for o in offsets)
    return (
        header
        + _scan_info(readbacks, detectors, detector_descriptions)
        + _scan_data(readbacks, detectors)
    )


def _extra_pvs() -> bytes:
    return (
        _int(2)
        + _string('S:SRcurrentAI')
        + _string('Storage ring current')
        + _int(EpicsType.DBR_CTRL_DOUBLE)
        + _int(1)
        + _string('mA')
        + _double(102.5)
        + _string('2ide:sampleName')
        + _string('Sample name')
        + _int(EpicsType.DBR_STRING)
        + _string('test-sample')
    )


def _header(scan_number: int, dimensions: list[int], extra_pvs_offset: int) -> bytes:
    return (
        _float(1.4)
        + _int(scan_number)
        + _int(len(dimensions))
        + b''.join(_int(d) for d in dimensions)
        + _int(1)  # is_regular
        + _int(extra_pvs_offset)
    )


def _write_flat_mda(
    path: Path,
    xs: list[float],
    ys: list[float],
    *,
    current_point: int | None = None,
    num_positioners: int = 2,
) -> Path:
    """A rank-1 scan carrying x on positioner 0 and y on positioner 1."""
    readbacks = [xs, ys][:num_positioners]
    detectors = [[float(i) for i in range(len(xs))]]
    header_len = len(_header(1, [len(xs)], 0))
    body = _scan(readbacks, detectors, current_point=current_point)

    path.write_bytes(_header(1, [len(xs)], header_len + len(body)) + body + _extra_pvs())
    return path


def _write_nested_mda(
    path: Path,
    ys: list[float],
    xs_per_row: list[list[float]],
    *,
    current_point: int | None = None,
    hole_at: int | None = None,
    header_dimensions: list[int] | None = None,
) -> Path:
    """A rank-2 scan: the outer readback holds y, each lower scan holds x.

    Rows at and past ``current_point`` are written the way an aborted scan leaves
    them -- the offset slot stays zero and no lower scan is emitted for it.

    ``header_dimensions`` overrides what the file header claims, so a caller can
    present a deeper rank than the body carries. Nothing downstream of the header
    is reached once the rank is refused, and the offsets stay consistent because
    the same list sizes both the probe and the emitted header.
    """
    dimensions = header_dimensions or [len(ys), len(xs_per_row[0])]
    written = len(xs_per_row) if current_point is None else current_point
    header_len = len(_header(len(dimensions), dimensions, 0))
    # A real aborted file keeps the row that was in flight: its offset is a valid
    # one but the block behind it is short, so reading it raises EOFError. Only
    # current_point excludes it -- skipping zero offsets alone does not.

    outer_readbacks = [ys, [0.0] * len(ys)]
    outer_detectors = [[0.0] * len(ys)]
    placeholder = [0] * len(ys)
    outer_len = len(_scan(outer_readbacks, outer_detectors, rank=2, lower_offsets=placeholder))

    inner_blocks = [_scan([xs, [0.0] * len(xs)], [[0.0] * len(xs)]) for xs in xs_per_row[:written]]

    if written < len(xs_per_row):
        in_flight = _scan(
            [xs_per_row[written], [0.0] * len(xs_per_row[written])],
            [[0.0] * len(xs_per_row[written])],
        )
        inner_blocks.append(in_flight[: len(in_flight) // 2])

    offsets: list[int] = []
    cursor = header_len + outer_len

    for block in inner_blocks:
        offsets.append(cursor)
        cursor += len(block)

    offsets.extend([0] * (len(ys) - len(offsets)))

    if hole_at is not None:
        offsets[hole_at] = 0
    outer = _scan(
        outer_readbacks,
        outer_detectors,
        rank=2,
        lower_offsets=offsets,
        current_point=current_point,
    )
    body = outer + b''.join(inner_blocks)

    path.write_bytes(_header(len(dimensions), dimensions, cursor) + body + _extra_pvs())
    return path


@pytest.fixture
def flat_mda(tmp_path: Path) -> Path:
    return _write_flat_mda(
        tmp_path / 'flat.mda',
        xs=[1.0, 2.0, 3.0, 4.0],
        ys=[10.0, 20.0, 30.0, 40.0],
    )


@pytest.fixture
def nested_mda(tmp_path: Path) -> Path:
    return _write_nested_mda(
        tmp_path / 'nested.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )


def test_flat_header_and_scan_metadata(flat_mda: Path) -> None:
    mda_file = MDAFile.read(flat_mda)

    assert mda_file.header.scan_number == 1
    assert mda_file.header.data_rank == 1
    assert mda_file.header.dimensions == [4]
    assert mda_file.header.is_regular
    assert mda_file.header.has_extra_pvs

    assert mda_file.scan.header.num_requested_points == 4
    assert mda_file.scan.info.scan_name == 'scan1'
    assert mda_file.scan.info.num_positioners == 2
    assert mda_file.scan.info.num_detectors == 1
    assert mda_file.scan.info.num_triggers == 1
    assert mda_file.scan.info.positioner[0].name == 'positioner0'
    assert mda_file.scan.info.positioner[0].unit == 'um'
    assert not mda_file.scan.lower_scans


def test_flat_extra_pvs_round_trip_both_epics_types(flat_mda: Path) -> None:
    """A counted string and a DBR_CTRL_DOUBLE take different branches of _read_pv."""
    extra_pvs = MDAFile.read(flat_mda).extra_pvs

    assert [pv.name for pv in extra_pvs] == ['S:SRcurrentAI', '2ide:sampleName']

    current, sample = extra_pvs
    assert current.epics_type == EpicsType.DBR_CTRL_DOUBLE
    assert current.unit == 'mA'
    assert current.value == pytest.approx([102.5])

    assert sample.epics_type == EpicsType.DBR_STRING
    assert sample.value == 'test-sample'


def test_flat_scan_reader_scales_both_axes(flat_mda: Path) -> None:
    """The flat reader takes x from positioner 0 and y from positioner 1."""
    positions = MDAFlatScanPositionFileReader(scale_to_meters=MILLIMETER_M).read(flat_mda)

    assert len(positions) == 4
    assert [point.index for point in positions] == list(range(4))
    assert positions[0].x_m == pytest.approx(1.0 * MILLIMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)
    assert positions[3].x_m == pytest.approx(4.0 * MILLIMETER_M)
    assert positions[3].y_m == pytest.approx(40.0 * MILLIMETER_M)


def test_nested_reader_walks_lower_scans(nested_mda: Path) -> None:
    """Rank-2 files pair each outer y with a whole lower scan of x values."""
    positions = MDAPositionFileReader(scale_to_meters=MICROMETER_M).read(nested_mda)

    assert len(positions) == 6
    assert [point.index for point in positions] == list(range(6))

    xs = [point.x_m for point in positions]
    assert xs == pytest.approx([v * MICROMETER_M for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)])

    assert positions[0].y_m == pytest.approx(100.0 * MICROMETER_M)
    assert positions[2].y_m == pytest.approx(100.0 * MICROMETER_M)
    assert positions[3].y_m == pytest.approx(200.0 * MICROMETER_M)
    assert positions[5].x_m == pytest.approx(6.0 * MICROMETER_M)


def test_nested_reader_rejects_a_deeper_rank(tmp_path: Path) -> None:
    """A rank-3 file nests a raster under each outer point, not a row.

    Walked as rank 2, its lower scans pair one point apiece with the outer readback, so a
    scan of thousands of frames returns a handful of positions -- along whichever axis the
    outermost scan drove, in whatever units that stage reports. The fixture declares the
    deeper rank in its header only, which is all the guard reads before refusing.
    """
    mda = _write_nested_mda(
        tmp_path / 'rank3.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        header_dimensions=[2, 2, 3],
    )

    with pytest.raises(ProbePositionParseError, match='rank-3'):
        MDAPositionFileReader(scale_to_meters=MICROMETER_M).read(mda)


# --- aborted scans: current_point is the count of rows actually written


def test_flat_reader_drops_the_preallocated_tail(tmp_path: Path) -> None:
    """Past current_point the file holds zeros, not coordinates at the origin."""
    mda = _write_flat_mda(
        tmp_path / 'aborted.mda',
        xs=[1.0, 2.0, 3.0, 0.0, 0.0],
        ys=[10.0, 20.0, 30.0, 0.0, 0.0],
        current_point=3,
    )

    positions = MDAFlatScanPositionFileReader(scale_to_meters=MILLIMETER_M).read(mda)

    assert len(positions) == 3
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MILLIMETER_M for v in (1.0, 2.0, 3.0)]
    )
    assert all(point.x_m != 0.0 for point in positions)


def test_nested_reader_stops_at_the_last_written_row(tmp_path: Path) -> None:
    """Unwritten rows carry offset 0; seeking there would reparse the file header."""
    mda = _write_nested_mda(
        tmp_path / 'aborted.mda',
        ys=[100.0, 200.0, 300.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        current_point=2,
    )

    positions = MDAPositionFileReader(scale_to_meters=MICROMETER_M).read(mda)

    assert len(positions) == 4
    assert [point.index for point in positions] == [0, 1, 2, 3]
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MICROMETER_M for v in (1.0, 2.0, 3.0, 4.0)]
    )
    assert positions[3].y_m == pytest.approx(200.0 * MICROMETER_M)


def test_nested_reader_halts_at_a_hole_rather_than_skipping_it(tmp_path: Path) -> None:
    """A zero offset inside the written range means the file disagrees with itself.

    Continuing past it would pair every later x row with the wrong y, which is worse
    than returning fewer points, so the reader stops at the hole.
    """
    mda = _write_nested_mda(
        tmp_path / 'hole.mda',
        ys=[100.0, 200.0, 300.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        hole_at=1,
    )

    positions = MDAPositionFileReader(scale_to_meters=MICROMETER_M).read(mda)

    assert len(positions) == 2
    assert [point.x_m for point in positions] == pytest.approx(
        [v * MICROMETER_M for v in (1.0, 2.0)]
    )
    assert all(point.y_m == pytest.approx(100.0 * MICROMETER_M) for point in positions)


def test_flat_reader_rejects_a_scan_with_no_points(tmp_path: Path) -> None:
    mda = _write_flat_mda(tmp_path / 'empty.mda', xs=[0.0, 0.0], ys=[0.0, 0.0], current_point=0)

    with pytest.raises(ProbePositionParseError) as info:
        MDAFlatScanPositionFileReader(scale_to_meters=MILLIMETER_M).read(mda)

    assert '0 of 2' in str(info.value)


def test_nested_reader_rejects_a_scan_with_no_points(tmp_path: Path) -> None:
    mda = _write_nested_mda(
        tmp_path / 'empty.mda',
        ys=[100.0, 200.0],
        xs_per_row=[[1.0, 2.0], [3.0, 4.0]],
        current_point=0,
    )

    with pytest.raises(ProbePositionParseError):
        MDAPositionFileReader(scale_to_meters=MICROMETER_M).read(mda)


def test_flat_reader_rejects_a_single_positioner_file(tmp_path: Path) -> None:
    """Most files in a 19-ID-E directory carry one positioner and are not flat scans.

    Indexing readback_array[1] on those used to raise a bare IndexError.
    """
    mda = _write_flat_mda(
        tmp_path / 'one_axis.mda', xs=[1.0, 2.0], ys=[10.0, 20.0], num_positioners=1
    )

    with pytest.raises(ProbePositionParseError) as info:
        MDAFlatScanPositionFileReader(scale_to_meters=MILLIMETER_M).read(mda)

    assert 'positioner' in str(info.value)


def test_complete_scan_is_not_reported_as_aborted(flat_mda: Path) -> None:
    assert not MDAFile.read(flat_mda).is_aborted


def test_aborted_scan_is_reported(tmp_path: Path) -> None:
    mda = _write_flat_mda(tmp_path / 'aborted.mda', xs=[1.0, 0.0], ys=[10.0, 0.0], current_point=1)

    assert MDAFile.read(mda).is_aborted


def test_vendored_xdrlib_is_importable() -> None:
    """The fallback must ship; load_plugins() would otherwise skip MDA in silence."""
    assert hasattr(_xdrlib, 'Packer')
    assert hasattr(_xdrlib, 'Unpacker')


def test_vendored_xdrlib_parses_identically(
    flat_mda: Path, nested_mda: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the Python 3.13 code path on whatever interpreter is running.

    ``xdrlib`` is a module global in mda_position_file, so swapping the attribute
    redirects every unpacker in the file without touching sys.modules.
    """
    reader = MDAFlatScanPositionFileReader(scale_to_meters=MILLIMETER_M)
    nested_reader = MDAPositionFileReader(scale_to_meters=MICROMETER_M)

    def coordinates() -> list[tuple[int, float, float]]:
        return [
            (point.index, point.x_m, point.y_m)
            for point in (*reader.read(flat_mda), *nested_reader.read(nested_mda))
        ]

    with_ambient = coordinates()
    monkeypatch.setattr(mda_position_file, 'xdrlib', _xdrlib)

    assert coordinates() == with_ambient
    assert mda_position_file.xdrlib is _xdrlib


_COLD_IMPORT_CHILD = """
import json
import sys


class BlockXdrlib:
    def find_spec(self, name, path=None, target=None):
        if name == 'xdrlib':
            raise ModuleNotFoundError("No module named 'xdrlib'", name=name)

        return None


sys.meta_path.insert(0, BlockXdrlib())

import ptychodus.plugins.mda as package
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.mda import mda_position_file

registry = PluginRegistry()
package.register_plugins(registry)

json.dump(
    {
        'xdrlib': mda_position_file.xdrlib.__name__,
        'readers': [plugin.simple_name for plugin in registry.probe_position_file_readers],
    },
    sys.stdout,
)
"""


def test_fallback_survives_the_partially_initialized_package() -> None:
    """Reproduce the real Python 3.13 import order in a cold interpreter.

    load_plugins() imports the package, whose ``__init__`` imports
    mda_position_file, which -- with no stdlib xdrlib -- imports ``_xdrlib`` back
    out of that same, still-executing package. It resolves because CPython falls
    back to importing the submodule when the attribute is not yet bound, but the
    cycle is real and is invisible on any interpreter that still ships xdrlib.
    """
    completed = subprocess.run(
        [sys.executable, '-c', _COLD_IMPORT_CHILD],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    result = json.loads(completed.stdout)
    assert result['xdrlib'] == 'ptychodus.plugins.mda._xdrlib'
    assert MDA_READER_NAMES <= set(result['readers'])


def test_mda_package_is_visible_to_the_plugin_scanner() -> None:
    """The vendored module lives inside the package, so only the package is scanned.

    load_plugins() walks the top of ptychodus.plugins with pkgutil.iter_modules,
    which yields a directory only when it has an __init__.py. That file is what
    makes the MDA readers register, and its absence would drop them silently.
    """
    names = {
        module.name
        for module in pkgutil.iter_modules(
            ptychodus.plugins.__path__, ptychodus.plugins.__name__ + '.'
        )
    }

    assert 'ptychodus.plugins.mda' in names
    assert 'ptychodus.plugins.mda._xdrlib' not in names


def test_every_mda_reader_registers() -> None:
    """A missing vendored xdrlib is a caught ModuleNotFoundError, not a test failure.

    load_plugins() logs and skips a plugin it cannot import, by design, so nothing
    else in the suite notices when these readers disappear.
    """
    registry = PluginRegistry.load_plugins()
    names = {plugin.simple_name for plugin in registry.probe_position_file_readers}

    assert MDA_READER_NAMES <= names


# --- detector-channel positions (APS 19-ID-E In-situ Nanoprobe) ----------------------
#
# An ISN fly scan drives one trajectory positioner and records both axis encoders as
# detector channels. The descriptions below are the ones real 19idAERO files carry.


def _write_channel_mda(
    path: Path,
    *,
    positioner: list[float],
    channels: list[tuple[str, list[float]]],
) -> Path:
    """A rank-1 scan with a single positioner and the axes in detector channels."""
    readbacks = [positioner]
    detectors = [values for _, values in channels]
    descriptions = [description for description, _ in channels]
    header_len = len(_header(1, [len(positioner)], 0))
    body = _scan(readbacks, detectors, detector_descriptions=descriptions)

    path.write_bytes(_header(1, [len(positioner)], header_len + len(body)) + body + _extra_pvs())
    return path


def _isn_reader(**overrides: object) -> MDADetectorChannelPositionFileReader:
    kwargs: dict[str, object] = {
        'x_description': 'X Axis',
        'y_description': 'Piezo Y',
        'x_index_fallback': 1,
        'y_index_fallback': 0,
    }
    kwargs.update(overrides)
    return MDADetectorChannelPositionFileReader(MILLIMETER_M, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def isn_mda(tmp_path: Path) -> Path:
    return _write_channel_mda(
        tmp_path / 'isn.mda',
        positioner=[1.0, 2.0, 3.0],
        channels=[
            ('Piezo Y (all)', [10.0, 20.0, 30.0]),
            ('X Axis', [1.5, 2.5, 3.5]),
            ('Z Axis', [-1.0, -1.0, -1.0]),
        ],
    )


def test_channel_reader_selects_axes_by_description(isn_mda: Path) -> None:
    positions = _isn_reader().read(isn_mda)

    assert [point.index for point in positions] == [0, 1, 2]
    assert [pytest.approx(point.x_m) for point in positions] == [
        1.5 * MILLIMETER_M,
        2.5 * MILLIMETER_M,
        3.5 * MILLIMETER_M,
    ]
    assert [pytest.approx(point.y_m) for point in positions] == [
        10.0 * MILLIMETER_M,
        20.0 * MILLIMETER_M,
        30.0 * MILLIMETER_M,
    ]


def test_channel_reader_prefers_the_description_over_the_index(tmp_path: Path) -> None:
    """Reordered channels must still be read correctly; that is the point of matching."""
    mda = _write_channel_mda(
        tmp_path / 'reordered.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('Z Axis', [-1.0, -1.0]),
            ('Piezo Y (all)', [10.0, 20.0]),
            ('X Axis', [1.5, 2.5]),
        ],
    )
    positions = _isn_reader().read(mda)

    assert positions[0].x_m == pytest.approx(1.5 * MILLIMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)


def test_channel_reader_falls_back_to_indexes(tmp_path: Path) -> None:
    """A file whose descriptions were changed still reads through the pinned indexes."""
    mda = _write_channel_mda(
        tmp_path / 'undescribed.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('unlabelled 0', [10.0, 20.0]),
            ('unlabelled 1', [1.5, 2.5]),
        ],
    )
    positions = _isn_reader().read(mda)

    assert positions[0].x_m == pytest.approx(1.5 * MILLIMETER_M)
    assert positions[0].y_m == pytest.approx(10.0 * MILLIMETER_M)


def test_channel_reader_rejects_an_ambiguous_description(tmp_path: Path) -> None:
    """Two plausible channels is an error: picking the first would be a silent guess."""
    mda = _write_channel_mda(
        tmp_path / 'ambiguous.mda',
        positioner=[1.0, 2.0],
        channels=[
            ('Piezo Y (all)', [10.0, 20.0]),
            ('X Axis', [1.5, 2.5]),
            ('X Axis coarse', [9.0, 9.0]),
        ],
    )
    with pytest.raises(ProbePositionParseError, match='Cannot choose between them'):
        _isn_reader().read(mda)


def test_channel_reader_names_the_channels_when_nothing_matches(tmp_path: Path) -> None:
    mda = _write_channel_mda(
        tmp_path / 'nomatch.mda',
        positioner=[1.0, 2.0],
        channels=[('unlabelled 0', [10.0, 20.0])],
    )
    with pytest.raises(ProbePositionParseError, match='unlabelled 0'):
        _isn_reader(x_index_fallback=9, y_index_fallback=9).read(mda)


def test_flat_reader_cannot_read_a_single_positioner_fly_scan(isn_mda: Path) -> None:
    """Why the channel reader exists: the flat-scan reader needs two positioners.

    Registering the flat reader for ISN made it raise on every real file.
    """
    with pytest.raises(ProbePositionParseError, match='this reader needs 2'):
        MDAFlatScanPositionFileReader(MILLIMETER_M).read(isn_mda)
