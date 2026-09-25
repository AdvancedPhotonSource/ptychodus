from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Final, Generic, TypeVar
import logging
import sys
import typing
import yaml

try:
    # xdrlib removed from the standard library in Python 3.13
    import xdrlib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # Full module path, not `from . import _xdrlib`, so this file stays usable as
    # an entry point. On 3.13 this reaches back into the package that is still
    # executing our own import; it resolves because the import machinery falls
    # through to the submodule when the attribute is not bound yet. Only 3.13
    # takes this branch, so tests/test_mda_position_file.py pins it in a
    # subprocess with the stdlib module blocked.
    from ptychodus.plugins.mda import _xdrlib as xdrlib  # type: ignore[no-redef]

import numpy

from ptychodus.api.typing import RealArrayType
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePositionParseError,
    ProbePosition,
)

T = TypeVar('T')

logger = logging.getLogger(__name__)

# Stride for line-major scan indexes: a point on scan line `line`, at column `column`
# within that line, is numbered `line * RASTER_LINE_INDEX_STRIDE + column`.
#
# Flattening a raster by running count instead ties the numbering to how many points
# each side of the pairing happens to hold. A fly scan drives the positioner over more
# points per line than the detector records, so a running count numbers the same physical
# point differently in the two files and the association slides by the surplus, one line's
# worth at a time, without ever failing to find a match. Line-major numbering names the
# point rather than counting arrivals, so the two agree by construction: surplus commanded
# points simply go unclaimed, a short line claims fewer, and a detector line past the end
# of the positioner record falls outside the position range.
#
# The stride bounds the points per line, and a scan line of a million points is not
# reachable at any dwell time an instrument runs. Readers that adopt this numbering must
# share this value and reject a line that reaches it, since a longer line would run into
# the next line's numbers.
RASTER_LINE_INDEX_STRIDE: Final = 1_000_000


class EpicsType(IntEnum):
    DBR_STRING = 0
    DBR_SHORT = 1
    DBR_FLOAT = 2
    DBR_ENUM = 3
    DBR_CHAR = 4
    DBR_LONG = 5
    DBR_DOUBLE = 6
    DBR_STS_STRING = 7
    DBR_STS_SHORT = 8
    DBR_STS_FLOAT = 9
    DBR_STS_ENUM = 10
    DBR_STS_CHAR = 11
    DBR_STS_LONG = 12
    DBR_STS_DOUBLE = 13
    DBR_TIME_STRING = 14
    DBR_TIME_SHORT = 15
    DBR_TIME_FLOAT = 16
    DBR_TIME_ENUM = 17
    DBR_TIME_CHAR = 18
    DBR_TIME_LONG = 19
    DBR_TIME_DOUBLE = 20
    DBR_GR_STRING = 21
    DBR_GR_SHORT = 22
    DBR_GR_FLOAT = 23
    DBR_GR_ENUM = 24
    DBR_GR_CHAR = 25
    DBR_GR_LONG = 26
    DBR_GR_DOUBLE = 27
    DBR_CTRL_STRING = 28
    DBR_CTRL_SHORT = 29
    DBR_CTRL_FLOAT = 30
    DBR_CTRL_ENUM = 31
    DBR_CTRL_CHAR = 32
    DBR_CTRL_LONG = 33
    DBR_CTRL_DOUBLE = 34


def read_int_from_buffer(fp: typing.BinaryIO) -> int:
    unpacker = xdrlib.Unpacker(fp.read(4))
    return unpacker.unpack_int()


def read_float_from_buffer(fp: typing.BinaryIO) -> float:
    unpacker = xdrlib.Unpacker(fp.read(4))
    return unpacker.unpack_float()


def read_counted_string(unpacker: xdrlib.Unpacker) -> str:
    length = unpacker.unpack_int()
    return unpacker.unpack_string().decode() if length else str()


def read_counted_string_from_buffer(fp: typing.BinaryIO) -> str:
    length = read_int_from_buffer(fp)

    if length:
        sz = (length + 3) // 4 * 4 + 4
        unpacker = xdrlib.Unpacker(fp.read(sz))
        return unpacker.unpack_string().decode()

    return str()


@dataclass(frozen=True)
class MDAHeader:
    version: float
    scan_number: int
    dimensions: list[int]
    is_regular: bool
    extra_pvs_offset: int

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAHeader:
        unpacker = xdrlib.Unpacker(fp.read(12))
        version = unpacker.unpack_float()
        scan_number = unpacker.unpack_int()
        data_rank = unpacker.unpack_int()

        unpacker.reset(fp.read(4 * data_rank + 8))
        dimensions = unpacker.unpack_farray(data_rank, unpacker.unpack_int)
        is_regular = unpacker.unpack_bool()
        extra_pvs_offset = unpacker.unpack_int()

        return cls(version, scan_number, dimensions, is_regular, extra_pvs_offset)

    @property
    def data_rank(self) -> int:
        return len(self.dimensions)

    @property
    def has_extra_pvs(self) -> bool:
        return self.extra_pvs_offset > 0

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'version': self.version,
            'scan_number': self.scan_number,
            'dimensions': self.dimensions,
            'is_regular': self.is_regular,
            'extra_pvs_offset': self.extra_pvs_offset,
        }


@dataclass(frozen=True)
class MDAScanHeader:
    rank: int
    num_requested_points: int
    current_point: int
    lower_scan_offsets: list[int]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanHeader:
        unpacker = xdrlib.Unpacker(fp.read(12))
        rank = unpacker.unpack_int()
        npts = unpacker.unpack_int()
        cpt = unpacker.unpack_int()
        lower_scan_offsets: list[int] = list()

        if rank > 1:
            unpacker.reset(fp.read(4 * npts))
            lower_scan_offsets = unpacker.unpack_farray(npts, unpacker.unpack_int)

        return cls(rank, npts, cpt, lower_scan_offsets)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'rank': self.rank,
            'num_requested_points': self.num_requested_points,
            'current_point': self.current_point,
            'lower_scan_offsets': self.lower_scan_offsets,
        }


@dataclass(frozen=True)
class MDAScanPositionerInfo:
    number: int
    name: str
    description: str
    step_mode: str
    unit: str
    readback_name: str
    readback_description: str
    readback_unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanPositionerInfo:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        step_mode = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        readback_name = read_counted_string_from_buffer(fp)
        readback_description = read_counted_string_from_buffer(fp)
        readback_unit = read_counted_string_from_buffer(fp)

        return cls(
            number,
            name,
            description,
            step_mode,
            unit,
            readback_name,
            readback_description,
            readback_unit,
        )

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'number': self.number,
            'name': self.name,
            'description': self.description,
            'step_mode': self.step_mode,
            'unit': self.unit,
            'readback_name': self.readback_name,
            'readback_description': self.readback_description,
            'readback_unit': self.readback_unit,
        }


@dataclass(frozen=True)
class MDAScanDetectorInfo:
    number: int
    name: str
    description: str
    unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanDetectorInfo:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        return cls(number, name, description, unit)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'number': self.number,
            'name': self.name,
            'description': self.description,
            'unit': self.unit,
        }


@dataclass(frozen=True)
class MDAScanTriggerInfo:
    number: int
    name: str
    command: float

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanTriggerInfo:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        command = read_float_from_buffer(fp)
        return cls(number, name, command)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'number': self.number,
            'name': self.name,
            'command': self.command,
        }


@dataclass(frozen=True)
class MDAScanInfo:
    scan_name: str
    time_stamp: str
    positioner: list[MDAScanPositionerInfo]
    detector: list[MDAScanDetectorInfo]
    trigger: list[MDAScanTriggerInfo]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanInfo:
        scan_name = read_counted_string_from_buffer(fp)
        time_stamp = read_counted_string_from_buffer(fp)

        unpacker = xdrlib.Unpacker(fp.read(12))
        np = unpacker.unpack_int()
        nd = unpacker.unpack_int()
        nt = unpacker.unpack_int()

        positioner = [MDAScanPositionerInfo.read(fp) for p in range(np)]
        detector = [MDAScanDetectorInfo.read(fp) for d in range(nd)]
        trigger = [MDAScanTriggerInfo.read(fp) for t in range(nt)]

        return cls(scan_name, time_stamp, positioner, detector, trigger)

    @property
    def num_positioners(self) -> int:
        return len(self.positioner)

    @property
    def num_detectors(self) -> int:
        return len(self.detector)

    @property
    def num_triggers(self) -> int:
        return len(self.trigger)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'scan_name': self.scan_name,
            'time_stamp': self.time_stamp,
            'positioner': [pos.to_mapping() for pos in self.positioner],
            'detector': [det.to_mapping() for det in self.detector],
            'trigger': [tri.to_mapping() for tri in self.trigger],
        }


@dataclass(frozen=True)
class MDAScanData:
    readback_array: RealArrayType  # double, shape: np x current_point
    detector_array: RealArrayType  # float, shape: nd x current_point

    @classmethod
    def read(
        cls, fp: typing.BinaryIO, scan_header: MDAScanHeader, scan_info: MDAScanInfo
    ) -> MDAScanData:
        npts = scan_header.num_requested_points
        np = scan_info.num_positioners
        nd = scan_info.num_detectors

        # A scan preallocates npts points but writes only current_point of them, so
        # an aborted scan leaves zeros in the tail that would otherwise read back as
        # real coordinates. Consume the full width to keep the stream aligned, then
        # keep what was acquired. Trim each row rather than slicing the stacked array:
        # with no detectors the array is 1-D and a two-axis slice would raise.
        cpt = max(0, min(scan_header.current_point, npts))

        unpacker = xdrlib.Unpacker(fp.read(8 * np * npts))
        readback_lol = [
            unpacker.unpack_farray(npts, unpacker.unpack_double)[:cpt] for p in range(np)
        ]
        readback_array = numpy.array(readback_lol)

        unpacker.reset(fp.read(4 * nd * npts))
        detector_lol = [
            unpacker.unpack_farray(npts, unpacker.unpack_float)[:cpt] for d in range(nd)
        ]
        detector_array = numpy.array(detector_lol)

        return cls(readback_array, detector_array)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'readback_array': f'{self.readback_array.dtype}{self.readback_array.shape}',
            'detector_array': f'{self.detector_array.dtype}{self.detector_array.shape}',
        }


@dataclass(frozen=True)
class MDAScan:
    header: MDAScanHeader
    info: MDAScanInfo
    data: MDAScanData
    lower_scans: list[MDAScan]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScan:
        header = MDAScanHeader.read(fp)
        info = MDAScanInfo.read(fp)
        data = MDAScanData.read(fp, header, info)
        lower_scans: list[MDAScan] = list()

        # Rows at and past current_point were never written: their offsets are zero,
        # and the row in progress when the scan stopped is short. Seeking to either
        # reads the file header back as a scan header. Stop at the first hole rather
        # than skipping it, since MDAPositionFileReader pairs lower scans with the
        # outer readback by position and a gap would shift every y after it.
        for offset in header.lower_scan_offsets[: header.current_point]:
            if offset <= 0:
                logger.warning(f'Lower scan {len(lower_scans)} is empty. Ignoring the rest.')
                break

            fp.seek(offset)
            scan = MDAScan.read(fp)
            lower_scans.append(scan)

        return cls(header, info, data, lower_scans)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'header': self.header.to_mapping(),
            'info': self.info.to_mapping(),
            'data': self.data.to_mapping(),
            'lower_scans': [scan.to_mapping() for scan in self.lower_scans],
        }

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)


@dataclass(frozen=True)
class MDAProcessVariable(Generic[T]):
    name: str
    description: str
    epics_type: EpicsType
    unit: str
    value: T

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'epicsType': self.epics_type.name,
            'unit': self.unit,
            'value': self.value,
        }


@dataclass(frozen=True)
class MDAFile:
    header: MDAHeader
    scan: MDAScan
    extra_pvs: list[MDAProcessVariable[Any]]

    @staticmethod
    def _read_pv(unpacker: xdrlib.Unpacker) -> MDAProcessVariable[typing.Any]:
        pv_name = read_counted_string(unpacker)
        pv_desc = read_counted_string(unpacker)
        pv_type = EpicsType(unpacker.unpack_int())

        if pv_type == EpicsType.DBR_STRING:
            value_str = read_counted_string(unpacker)
            return MDAProcessVariable[str](pv_name, pv_desc, pv_type, str(), value_str)

        count = unpacker.unpack_int()
        pv_unit = read_counted_string(unpacker)

        if pv_type == EpicsType.DBR_CTRL_CHAR:
            value_char = unpacker.unpack_fstring(count).decode()
            value_char = value_char.split('\x00', 1)[0]  # treat as null-terminated string
            return MDAProcessVariable[str](pv_name, pv_desc, pv_type, pv_unit, value_char)
        elif pv_type == EpicsType.DBR_CTRL_SHORT:
            value_short = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pv_name, pv_desc, pv_type, pv_unit, value_short)
        elif pv_type == EpicsType.DBR_CTRL_LONG:
            value_long = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pv_name, pv_desc, pv_type, pv_unit, value_long)
        elif pv_type == EpicsType.DBR_CTRL_FLOAT:
            value_float = unpacker.unpack_farray(count, unpacker.unpack_float)
            return MDAProcessVariable[list[float]](pv_name, pv_desc, pv_type, pv_unit, value_float)
        elif pv_type == EpicsType.DBR_CTRL_DOUBLE:
            value_double = unpacker.unpack_farray(count, unpacker.unpack_double)
            return MDAProcessVariable[list[float]](pv_name, pv_desc, pv_type, pv_unit, value_double)

        return MDAProcessVariable[str](pv_name, pv_desc, pv_type, pv_unit, str())

    @classmethod
    def read(cls, file_path: Path) -> MDAFile:
        extra_pvs: list[MDAProcessVariable[Any]] = list()

        with file_path.open(mode='rb') as fp:
            header = MDAHeader.read(fp)
            scan = MDAScan.read(fp)

            if header.has_extra_pvs:
                fp.seek(header.extra_pvs_offset)
                unpacker = xdrlib.Unpacker(fp.read())
                number_pvs = unpacker.unpack_int()

                for pvidx in range(number_pvs):
                    pv = cls._read_pv(unpacker)
                    extra_pvs.append(pv)

        if scan.header.current_point < scan.header.num_requested_points:
            logger.warning(
                f'"{file_path}" is an aborted scan:'
                f' {scan.header.current_point} of {scan.header.num_requested_points}'
                ' points were acquired.'
            )

        return cls(header, scan, extra_pvs)

    @property
    def is_aborted(self) -> bool:
        return self.scan.header.current_point < self.scan.header.num_requested_points

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'header': self.header.to_mapping(),
            'scan': self.scan.to_mapping(),
            'extra_pvs': [pv.to_mapping() for pv in self.extra_pvs],
        }

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)


def _require_positioners(mda_file: MDAFile, count: int, file_path: Path) -> RealArrayType:
    """Return the readback array, or explain why it cannot supply `count` axes."""
    readback_array = mda_file.scan.data.readback_array

    if readback_array.ndim != 2 or readback_array.shape[0] < count:
        raise ProbePositionParseError(
            f'"{file_path}" has {mda_file.scan.info.num_positioners} positioner(s);'
            f' this reader needs {count}.'
        )

    return readback_array


def _require_points(
    point_list: list[ProbePosition], mda_file: MDAFile, file_path: Path
) -> ProbePositionSequence:
    """An aborted scan can leave nothing behind; say so rather than returning empty."""
    if not point_list:
        raise ProbePositionParseError(
            f'No probe positions in "{file_path}":'
            f' {mda_file.scan.header.current_point} of'
            f' {mda_file.scan.header.num_requested_points} points were acquired.'
        )

    return ProbePositionSequence(point_list)


class MDAPositionFileReader(ProbePositionFileReader):
    """Read scan positions from a two-dimensional raster: one positioner per axis.

    The outer scan supplies y and each of its lower scans supplies that row's x.

    `line_index_stride` selects how points are numbered. Left at None they are numbered
    by running count over the whole raster. Set to RASTER_LINE_INDEX_STRIDE they are
    numbered line-major, which is what an instrument needs whose detector records fewer
    points per line than the positioner is driven over; see that constant. A line holding
    at least the stride is rejected rather than allowed to collide with the next line.
    """

    def __init__(self, scale_to_meters: float, *, line_index_stride: int | None = None) -> None:
        self._scale_to_meters = scale_to_meters
        self._line_index_stride = line_index_stride

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        mda_file = MDAFile.read(file_path)
        data_rank = mda_file.header.data_rank

        # A deeper file nests another scan under each row, so its lower scans hold
        # rasters rather than rows. Pairing them with the outer readback would return one
        # point per raster -- a handful of positions for a scan of thousands of frames,
        # silently, and along whichever axis the outermost scan happened to drive.
        if data_rank > 2:
            raise ProbePositionParseError(
                f'"{file_path}" is a rank-{data_rank} scan of {mda_file.header.dimensions};'
                ' this reader needs a two-dimensional raster.'
            )

        yscan = mda_file.scan
        yarray = _require_positioners(mda_file, 1, file_path)[0, :]

        stride = self._line_index_stride

        for line, (y, xscan) in enumerate(zip(yarray, yscan.lower_scans)):
            xarray = xscan.data.readback_array[0, :]

            if stride is not None and len(xarray) >= stride:
                raise ProbePositionParseError(
                    f'Line {line} of "{file_path}" holds {len(xarray)} points, which reaches'
                    f' the line-major index stride of {stride}; its numbering would collide'
                    ' with the next line.'
                )

            for column, x in enumerate(xarray):
                point = ProbePosition(
                    index=len(point_list) if stride is None else line * stride + column,
                    x_m=float(x) * self._scale_to_meters,
                    y_m=float(y) * self._scale_to_meters,
                )
                point_list.append(point)

        return _require_points(point_list, mda_file, file_path)


class MDAFlatScanPositionFileReader(ProbePositionFileReader):
    def __init__(self, scale_to_meters: float) -> None:
        self._scale_to_meters = scale_to_meters

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        mda_file = MDAFile.read(file_path)
        readback_array = _require_positioners(mda_file, 2, file_path)
        xarray = readback_array[0, :]
        yarray = readback_array[1, :]

        for idx, (x, y) in enumerate(zip(xarray, yarray)):
            point = ProbePosition(
                index=idx,
                x_m=float(x) * self._scale_to_meters,
                y_m=float(y) * self._scale_to_meters,
            )
            point_list.append(point)

        return _require_points(point_list, mda_file, file_path)


def _require_detectors(mda_file: MDAFile, count: int, file_path: Path) -> RealArrayType:
    """Return the detector array, or explain why it cannot supply `count` channels."""
    detector_array = mda_file.scan.data.detector_array

    if detector_array.ndim != 2 or detector_array.shape[0] < count:
        raise ProbePositionParseError(
            f'"{file_path}" has {mda_file.scan.info.num_detectors} detector channel(s);'
            f' this reader needs {count}.'
        )

    return detector_array


def _select_channel(
    mda_file: MDAFile,
    description: str,
    index_fallback: int,
    axis: str,
    file_path: Path,
) -> int:
    """Index of the detector channel holding one axis, by description then by position.

    Matching the description survives a channel list that grows or is reordered, which
    an index cannot. An ambiguous description is an error rather than a guess: picking
    the first of several matches is how a reconstruction ends up silently built on the
    wrong axis.
    """
    detectors = mda_file.scan.info.detector
    wanted = description.casefold()
    matches = [i for i, d in enumerate(detectors) if wanted in d.description.casefold()]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        named = ', '.join(f'd[{i}] {detectors[i].description!r}' for i in matches)
        raise ProbePositionParseError(
            f'"{file_path}" has {len(matches)} detector channels matching {description!r}'
            f' for the {axis} axis: {named}. Cannot choose between them.'
        )

    if index_fallback < len(detectors):
        logger.debug(
            'No detector channel in "%s" describes %r; falling back to d[%d] %r for %s.',
            file_path,
            description,
            index_fallback,
            detectors[index_fallback].description,
            axis,
        )
        return index_fallback

    named = ', '.join(f'd[{i}] {d.description!r}' for i, d in enumerate(detectors))
    raise ProbePositionParseError(
        f'"{file_path}" has no detector channel describing {description!r} for the {axis}'
        f' axis, and no d[{index_fallback}] to fall back to. Channels present: {named}.'
    )


class MDADetectorChannelPositionFileReader(ProbePositionFileReader):
    """Read scan positions from MDA detector channels rather than from positioners.

    A fly scan drives a single trajectory positioner and records each axis encoder as a
    detector channel, so :class:`MDAFlatScanPositionFileReader`, which requires two
    positioners, cannot read one. APS 19-ID-E In-situ Nanoprobe files have exactly that
    shape: rank 1, one positioner (``19idAERO:m2.VAL``, the X setpoint) and 28 detector
    channels, among them ``19idAERO:m2.RBV`` ("X Axis") and ``19idAERO:SM1.RBV``
    ("Piezo Y (all)") -- the encoder readbacks the reconstruction needs.

    The readbacks are preferred over the setpoint: a fly scan does not stop on the
    demanded coordinate, so the setpoint describes the trajectory rather than where each
    frame was taken.
    """

    def __init__(
        self,
        scale_to_meters: float,
        *,
        x_description: str,
        y_description: str,
        x_index_fallback: int,
        y_index_fallback: int,
    ) -> None:
        self._scale_to_meters = scale_to_meters
        self._x_description = x_description
        self._y_description = y_description
        self._x_index_fallback = x_index_fallback
        self._y_index_fallback = y_index_fallback

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        mda_file = MDAFile.read(file_path)
        x_index = _select_channel(
            mda_file, self._x_description, self._x_index_fallback, 'x', file_path
        )
        y_index = _select_channel(
            mda_file, self._y_description, self._y_index_fallback, 'y', file_path
        )
        detector_array = _require_detectors(mda_file, 1 + max(x_index, y_index), file_path)

        for idx, (x, y) in enumerate(zip(detector_array[x_index, :], detector_array[y_index, :])):
            point = ProbePosition(
                index=idx,
                x_m=float(x) * self._scale_to_meters,
                y_m=float(y) * self._scale_to_meters,
            )
            point_list.append(point)

        return _require_points(point_list, mda_file, file_path)


if __name__ == '__main__':
    file_path = Path(sys.argv[1])
    mda_file = MDAFile.read(file_path)
    print(mda_file)
