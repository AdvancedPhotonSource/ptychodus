"""Synthetic EPICS MDA files, written as XDR with :mod:`struct`.

MDA is a binary XDR container, and ``tests/data`` is otherwise text-only, so these files
are synthesized in ``tmp_path``. The writer emits XDR directly rather than through
``xdrlib``, which keeps it an independent oracle for the reader instead of a round-trip
through the same code.

Three shapes are written: a flat rank-1 scan with x and y on two positioners; a rank-2
scan whose outer readback holds y while each lower scan holds x; and scans whose axes are
detector channels, at rank 1 (one trajectory positioner, as 19-ID-E ISN fly scans) or at
rank 2 (a channel row per lower scan, as 26-ID-C rasters).
"""

from __future__ import annotations

from pathlib import Path
import struct

from ptychodus.plugins.mda.mda_position_file import EpicsType


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
    detector_names: list[str] | None = None,
) -> bytes:
    descriptions = detector_descriptions or [None] * len(detectors)  # type: ignore[list-item]
    names = detector_names or [f'detector{i}' for i in range(len(detectors))]
    return (
        _string('scan1')
        + _string('Jun 19, 2025 13:45:00.000000')
        + _int(len(readbacks))
        + _int(len(detectors))
        + _int(1)
        + b''.join(_positioner(i, f'positioner{i}') for i in range(len(readbacks)))
        + b''.join(_detector(i, names[i], descriptions[i]) for i in range(len(detectors)))
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
    detector_names: list[str] | None = None,
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
        + _scan_info(readbacks, detectors, detector_descriptions, detector_names)
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


def write_flat_mda(
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


def write_nested_mda(
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


def write_channel_mda(
    path: Path,
    *,
    positioner: list[float],
    channels: list[tuple[str, list[float]]],
    channel_names: list[str] | None = None,
    current_point: int | None = None,
) -> Path:
    """A rank-1 scan with a single positioner and the axes in detector channels.

    Each channel is a (description, values) pair; `channel_names` gives their PV names,
    which default to ``detector<i>``.
    """
    readbacks = [positioner]
    detectors = [values for _, values in channels]
    descriptions = [description for description, _ in channels]
    header_len = len(_header(1, [len(positioner)], 0))
    body = _scan(
        readbacks,
        detectors,
        current_point=current_point,
        detector_descriptions=descriptions,
        detector_names=channel_names,
    )

    path.write_bytes(_header(1, [len(positioner)], header_len + len(body)) + body + _extra_pvs())
    return path


def write_nested_channel_mda(
    path: Path,
    *,
    outer: list[float],
    channel_names: list[str],
    rows: list[list[list[float]]],
) -> Path:
    """A rank-2 scan whose lower scans record the named detector channels, row by row.

    `outer` is the outer positioner's readback, one value per row. ``rows[i][c]`` holds
    channel ``c``'s values along row ``i``; every row's lower scan drives a positioner of
    its own, which these files do not read.
    """
    num_columns = len(rows[0][0])
    dimensions = [len(outer), num_columns]
    header_len = len(_header(len(dimensions), dimensions, 0))
    descriptions = [f'{name} description' for name in channel_names]

    outer_readbacks = [outer]
    outer_detectors = [[0.0] * len(outer)]
    placeholder = [0] * len(outer)
    outer_len = len(_scan(outer_readbacks, outer_detectors, rank=2, lower_offsets=placeholder))

    inner_blocks = [
        _scan(
            [[float(i) for i in range(num_columns)]],
            row,
            detector_descriptions=descriptions,
            detector_names=channel_names,
        )
        for row in rows
    ]
    offsets: list[int] = []
    cursor = header_len + outer_len

    for block in inner_blocks:
        offsets.append(cursor)
        cursor += len(block)

    outer_block = _scan(outer_readbacks, outer_detectors, rank=2, lower_offsets=offsets)
    body = outer_block + b''.join(inner_blocks)
    path.write_bytes(_header(len(dimensions), dimensions, cursor) + body + _extra_pvs())
    return path
