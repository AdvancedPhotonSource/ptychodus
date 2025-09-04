from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Generic, TypeVar
import logging
import sys
import typing
import yaml

try:
    # xdrlib removed from the standard library in Python 3.13
    import xdrlib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    # use full module path to make this file usable as an entry point
    from ptychodus.plugins._xdrlib import xdrlib  # type: ignore[no-redef]

import numpy
import numpy.typing

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

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
    readback_array: numpy.typing.NDArray[numpy.floating[Any]]  # double, shape: np x npts
    detector_array: numpy.typing.NDArray[numpy.floating[Any]]  # float, shape: nd x npts

    @classmethod
    def read(
        cls, fp: typing.BinaryIO, scan_header: MDAScanHeader, scan_info: MDAScanInfo
    ) -> MDAScanData:
        npts = scan_header.num_requested_points
        np = scan_info.num_positioners
        nd = scan_info.num_detectors

        unpacker = xdrlib.Unpacker(fp.read(8 * np * npts))
        readback_lol = [unpacker.unpack_farray(npts, unpacker.unpack_double) for p in range(np)]
        readback_array = numpy.array(readback_lol)

        unpacker.reset(fp.read(4 * nd * npts))
        detector_lol = [unpacker.unpack_farray(npts, unpacker.unpack_float) for d in range(nd)]
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

        for offset in header.lower_scan_offsets:
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

        return cls(header, scan, extra_pvs)

    def to_mapping(self) -> Mapping[str, Any]:
        return {
            'header': self.header.to_mapping(),
            'scan': self.scan.to_mapping(),
            'extra_pvs': [pv.to_mapping() for pv in self.extra_pvs],
        }

    def __str__(self) -> str:
        return yaml.safe_dump(self.to_mapping(), sort_keys=False)


class MDAPositionFileReader(PositionFileReader):
    def __init__(self, scale_to_meters: float) -> None:
        self._scale_to_meters = scale_to_meters

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        mda_file = MDAFile.read(file_path)

        yscan = mda_file.scan
        yarray = yscan.data.readback_array[0, :]

        for y, xscan in zip(yarray, yscan.lower_scans):
            xarray = xscan.data.readback_array[0, :]

            for x in xarray:
                point = ScanPoint(
                    index=len(point_list),
                    position_x_m=float(x) * self._scale_to_meters,
                    position_y_m=float(y) * self._scale_to_meters,
                )
                point_list.append(point)

        return PositionSequence(point_list)


class MDAFlatScanPositionFileReader(PositionFileReader):
    def __init__(self, scale_to_meters: float) -> None:
        self._scale_to_meters = scale_to_meters

    def read(self, file_path: Path) -> PositionSequence:
        point_list = list()

        mda_file = MDAFile.read(file_path)
        xarray = mda_file.scan.data.readback_array[0, :]
        yarray = mda_file.scan.data.readback_array[1, :]

        for idx, (x, y) in enumerate(zip(xarray, yarray)):
            point = ScanPoint(
                index=idx,
                position_x_m=float(x) * self._scale_to_meters,
                position_y_m=float(y) * self._scale_to_meters,
            )
            point_list.append(point)

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='MDA',
        display_name='EPICS MDA Files (*.mda)',
    )
    registry.position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_2IDD',
        display_name='APS 2-ID-D Microprobe Files (*.mda)',
    )
    registry.position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_2IDE',
        display_name='APS 2-ID-E Microprobe Files (*.mda)',
    )
    registry.position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.h5 *.hdf5)',
    )
    registry.position_file_readers.register_plugin(
        MDAFlatScanPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_ISN_MDA',
        display_name='APS 19-ID In-Situ Nanoprobe Files (*.mda)',
    )
    registry.position_file_readers.register_plugin(
        MDAFlatScanPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='CNM_APS_HXN',
        display_name='CNM/APS Hard X-ray Nanoprobe Files (*.mda)',
    )


if __name__ == '__main__':
    file_path = Path(sys.argv[1])
    mda_file = MDAFile.read(file_path)
    print(mda_file)
