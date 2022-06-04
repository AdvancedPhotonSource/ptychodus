from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Generic, TypeVar
import xdrlib

T = TypeVar('T')


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
    pass # FIXME


class MDAExtraPVType(IntEnum):
    DBR_STRING = 0
    DBR_CTRL_SHORT = 29
    DBR_CTRL_FLOAT = 30
    DBR_CTRL_CHAR = 32
    DBR_CTRL_LONG = 33
    DBR_CTRL_DOUBLE = 34


@dataclass(frozen=True)
class MDAProcessVariable(Generic[T]):
    name: str
    description: str
    unit: str
    values: list[T]


@dataclass(frozen=True)
class MDAFile:
    header: MDAHeaderSection
    scan: MDAScanSection
    extra_pvs: list[MDAProcessVariable]

    @staticmethod
    def _read_counted_string(unpacker: xdrlib.Unpacker) -> str:
        length = unpacker.unpack_int()
        return unpacker.unpack_string().decode() if length else str()

    @staticmethod
    def _read_pv(unpacker: xdrlib.Unpacker) -> MDAProcessVariable[Any]:
        pvName = MDAFile._read_counted_string(unpacker)
        pvDesc = MDAFile._read_counted_string(unpacker)
        pvType = MDAExtraPVType(unpacker.unpack_int())

# FIXME BEGIN
        unit = ''
        value = ''
        count = 0
        if type != 0:   # not DBR_STRING
            count = u.unpack_int()  #
            n = u.unpack_int()      # length of unit string
            if n: unit = u.unpack_string().decode()

        if type == 0: # DBR_STRING
            n = u.unpack_int()      # length of value string
            if n: value = u.unpack_string().decode()
        elif type == 32: # DBR_CTRL_CHAR
            #value = u.unpack_fstring(count)
            v = u.unpack_farray(count, u.unpack_int)
            value = ""
            for i in range(len(v)):
                # treat the byte array as a null-terminated string
                if v[i] == 0: break
                value = value + chr(v[i])

        elif type == 29: # DBR_CTRL_SHORT
            value = u.unpack_farray(count, u.unpack_int)
        elif type == 33: # DBR_CTRL_LONG
            value = u.unpack_farray(count, u.unpack_int)
        elif type == 30: # DBR_CTRL_FLOAT
            value = u.unpack_farray(count, u.unpack_float)
        elif type == 34: # DBR_CTRL_DOUBLE
            value = u.unpack_farray(count, u.unpack_double)

        dict[name] = (desc, unit, value)
# FIXME END

        # FIXME return MDAProcessVariable[X]()

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
                        pv = MDAFile._read_pv(unpacker)
                        extra_pvs.append(pv)
        except OSError as exc:
            logger.exception(exc)

        return cls(header, scan, extra_pvs)

