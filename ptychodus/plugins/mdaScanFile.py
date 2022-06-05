from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Generic, TypeVar
import sys
import typing
import xdrlib


T = TypeVar('T')

logger = logging.getLogger(__name__)


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
        sz = (length + 3)//4*4 + 4
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
        return (self.extra_pvs_offset > 0)


@dataclass(frozen=True)
class MDAScanHeader:
    rank: int
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

        return cls(rank, cpt, lower_scan_offsets)

    @property
    def number_of_requested_points(self) -> int:
        return len(self.lower_scan_offsets)


@dataclass(frozen=True)
class MDAScanPositioner:
    number: int
    name: str
    description: str
    step_mode: str
    unit: str
    readback_name: str
    readback_description: str
    readback_unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanPositioner:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        step_mode = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        readback_name = read_counted_string_from_buffer(fp)
        readback_description = read_counted_string_from_buffer(fp)
        readback_unit = read_counted_string_from_buffer(fp)

        return cls(number, name, description, step_mode, unit, readback_name,
                readback_description, readback_unit)


@dataclass(frozen=True)
class MDAScanDetector:
    number: int
    name: str
    description: str
    unit: str

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanDetector:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        description = read_counted_string_from_buffer(fp)
        unit = read_counted_string_from_buffer(fp)
        return cls(number, name, description, unit)


@dataclass(frozen=True)
class MDAScanTrigger:
    number: int
    name: str
    command: float

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanTrigger:
        number = read_int_from_buffer(fp)
        name = read_counted_string_from_buffer(fp)
        command = read_float_from_buffer(fp)
        return cls(number, name, command)


@dataclass(frozen=True)
class MDAScanInfo:
    scan_name: str
    time_stamp: str
    positioners: list[MDAScanPositioner]
    detectors: list[MDAScanDetector]
    triggers: list[MDAScanTrigger]

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanInfo:
        scan_name = read_counted_string_from_buffer(fp)
        time_stamp = read_counted_string_from_buffer(fp)

        unpacker = xdrlib.Unpacker(fp.read(12))
        np = unpacker.unpack_int()
        nd = unpacker.unpack_int()
        nt = unpacker.unpack_int()

        positioners = [MDAScanPositioner.read(fp) for p in range(np)]
        detectors = [MDAScanDetector.read(fp) for d in range(nd)]
        triggers = [MDAScanTrigger.read(fp) for t in range(nt)]

        return cls(scan_name, time_stamp, positioners, detectors, triggers)


@dataclass(frozen=True)
class MDAScanData:
    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanData:
        return cls() # TODO


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
        data = MDAScanData.read(fp)
        lower_scans: list[MDAScan] = list()

        for offset in header.lower_scan_offsets:
            fp.seek(offset)
            scan = MDAScan.read(fp)
            lower_scans.append(scan)

        return cls(header, info, data, lower_scans)


@dataclass(frozen=True)
class MDAProcessVariable(Generic[T]):
    name: str
    description: str
    epicsType: EpicsType
    unit: str
    value: T


@dataclass(frozen=True)
class MDAFile:
    header: MDAHeader
    scan: MDAScan
    extra_pvs: list[MDAProcessVariable]

    @staticmethod
    def _read_pv(unpacker: xdrlib.Unpacker) -> MDAProcessVariable[typing.Any]:
        pvName = read_counted_string(unpacker)
        pvDesc = read_counted_string(unpacker)
        pvType = EpicsType(unpacker.unpack_int())

        if pvType == EpicsType.DBR_STRING:
            valueStr = read_counted_string(unpacker)
            return MDAProcessVariable[str](pvName, pvDesc, pvType, str(), valueStr)

        count = unpacker.unpack_int()
        pvUnit = read_counted_string(unpacker)

        if pvType == EpicsType.DBR_CTRL_CHAR:
            valueChar = unpacker.unpack_fstring(count).decode()
            valueChar = valueChar.split('\x00', 1)[0]  # treat as null-terminated string
            return MDAProcessVariable[str](pvName, pvDesc, pvType, pvUnit, valueChar)
        elif pvType == EpicsType.DBR_CTRL_SHORT:
            valueShort = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pvName, pvDesc, pvType, pvUnit, valueShort)
        elif pvType == EpicsType.DBR_CTRL_LONG:
            valueLong = unpacker.unpack_farray(count, unpacker.unpack_int)
            return MDAProcessVariable[list[int]](pvName, pvDesc, pvType, pvUnit, valueLong)
        elif pvType == EpicsType.DBR_CTRL_FLOAT:
            valueFloat = unpacker.unpack_farray(count, unpacker.unpack_float)
            return MDAProcessVariable[list[float]](pvName, pvDesc, pvType, pvUnit, valueFloat)
        elif pvType == EpicsType.DBR_CTRL_DOUBLE:
            valueDouble = unpacker.unpack_farray(count, unpacker.unpack_double)
            return MDAProcessVariable[list[float]](pvName, pvDesc, pvType, pvUnit, valueDouble)

        return MDAProcessVariable[str](pvName, pvDesc, pvType, pvUnit, str())

    @classmethod
    def read(cls, filePath: Path) -> MDAFile:
        extra_pvs: list[MDAProcessVariable] = list()

        try:
            with filePath.open(mode='rb') as fp:
                header = MDAHeader.read(fp)
                scan = MDAScan.read(fp)

                if header.has_extra_pvs:
                    fp.seek(header.extra_pvs_offset)
                    unpacker = xdrlib.Unpacker(fp.read())
                    number_pvs = unpacker.unpack_int()

                    for pvidx in range(number_pvs):
                        pv = cls._read_pv(unpacker)
                        extra_pvs.append(pv)
        except OSError as exc:
            logger.exception(exc)

        return cls(header, scan, extra_pvs)


if __name__ == '__main__':
    filePath = Path(sys.argv[1])
    mda = MDAFile.read(filePath)
    print(mda)
