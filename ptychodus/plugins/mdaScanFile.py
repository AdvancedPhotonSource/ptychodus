from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Generic, TypeVar
import sys
import xdrlib

T = TypeVar('T')


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


def read_counted_string(unpacker: xdrlib.Unpacker) -> str:
    length = unpacker.unpack_int()
    return unpacker.unpack_string().decode() if length else str()


@dataclass(frozen=True)
class MDAHeaderSection:
    version: float
    scan_number: int
    dimensions: list[int]
    is_regular: bool
    extra_pvs_offset: int

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAHeaderSection:
        unpacker = xdrlib.Unpacker(fp.read(12))
        version = unpacker.unpack_float()
        scan_number = unpacker.unpack_int()
        data_rank = unpacker.unpack_int()

        unpacker = xdrlib.Unpacker(fp.read(4 * data_rank + 8))
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
class MDAScanSection:
    # contains the scan data
    pass  # FIXME

    @classmethod
    def read(cls, fp: typing.BinaryIO) -> MDAScanSection:
        return None  # FIXME


@dataclass(frozen=True)
class MDAProcessVariable(Generic[T]):
    name: str
    description: str
    epicsType: EpicsType
    unit: str
    value: T


@dataclass(frozen=True)
class MDAFile:
    header: MDAHeaderSection
    scan: MDAScanSection
    extra_pvs: list[MDAProcessVariable]

    @staticmethod
    def _read_pv(unpacker: xdrlib.Unpacker) -> MDAProcessVariable[Any]:
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
                header = MDAHeaderSection.read(fp)
                scan = MDAScanSection.read(fp)

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
