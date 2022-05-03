from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional
import csv
import logging

import h5py
import numpy

from ..h5DataFileReader import H5DataFileReader
from ptychodus.api.data import DatasetState
from ptychodus.api.scan import ScanFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataFile:
    name: str
    filePath: Path
    dataPath: str

    def getState(self) -> DatasetState:
        state = DatasetState.MISSING

        if self.filePath.is_file():
            state = DatasetState.FOUND

            try:
                with h5py.File(self.filePath, 'r') as h5File:
                    if self.dataPath in h5File:
                        state = DatasetState.VALID
            except OSError:
                pass

        return state


@dataclass(frozen=True)
class DataGroup:
    datafileList: list[DataFile] = field(default_factory=list)

    def __iter__(self):
        return iter(self.datafileList)

    def __getitem__(self, index: int) -> DataFile:
        return self.datafileList[index]

    def __len__(self) -> int:
        return len(self.datafileList)


@dataclass(frozen=True)
class DetectorSpecificGroup:
    photon_energy_eV: float
    x_pixels_in_detector: int
    y_pixels_in_detector: int


@dataclass(frozen=True)
class DetectorGroup:
    detectorSpecific: DetectorSpecificGroup
    detector_distance_m: float
    beam_center_x_px: int
    beam_center_y_px: int
    bit_depth_image: int
    x_pixel_size_m: float
    y_pixel_size_m: float


@dataclass(frozen=True)
class InstrumentGroup:
    detector: DetectorGroup


@dataclass(frozen=True)
class GoniometerGroup:
    chi_deg: float

    @property
    def chi_rad(self) -> float:
        return numpy.deg2rad(self.chi_deg)


@dataclass(frozen=True)
class SampleGroup:
    goniometer: GoniometerGroup


@dataclass(frozen=True)
class EntryGroup:
    data: Optional[DataGroup]
    instrument: Optional[InstrumentGroup]
    sample: Optional[SampleGroup]


class VelociprobeDataFileReader(H5DataFileReader):
    def __init__(self) -> None:
        super().__init__(simpleName='Velociprobe',
                         fileFilter='Velociprobe Master Files (*.h5 *.hdf5)')
        self.masterFilePath: Optional[Path] = None
        self.entryGroup: Optional[EntryGroup] = None

    def _readDataGroup(self, h5DataGroup: Optional[h5py.Group]) -> Optional[DataGroup]:
        if h5DataGroup is None:
            return None

        datafileList = list()

        for name, h5Item in h5DataGroup.items():
            h5Item = h5DataGroup.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink) and self.masterFilePath is not None:
                datafile = DataFile(name=name,
                                    filePath=self.masterFilePath.parent / h5Item.filename,
                                    dataPath=str(h5Item.path))
                datafileList.append(datafile)

        datafileList.sort(key=lambda x: x.name)

        return DataGroup(datafileList)

    @staticmethod
    def _readInstrumentGroup(h5InstrumentGroup: Optional[h5py.Group]) -> Optional[InstrumentGroup]:
        if h5InstrumentGroup is None:
            return None

        h5DetectorGroup = h5InstrumentGroup['detector']
        h5DetectorSpecificGroup = h5DetectorGroup['detectorSpecific']

        h5PhotonEnergy = h5DetectorSpecificGroup['photon_energy']
        assert h5PhotonEnergy.attrs['units'] == b'eV'
        h5XPixelsInDetector = h5DetectorSpecificGroup['x_pixels_in_detector']
        h5YPixelsInDetector = h5DetectorSpecificGroup['y_pixels_in_detector']

        detectorSpecific = DetectorSpecificGroup(photon_energy_eV=h5PhotonEnergy[()],
                                                 x_pixels_in_detector=h5XPixelsInDetector[()],
                                                 y_pixels_in_detector=h5YPixelsInDetector[()])

        h5DetectorDistance = h5DetectorGroup['detector_distance']
        assert h5DetectorDistance.attrs['units'] == b'm'
        h5BeamCenterX = h5DetectorGroup['beam_center_x']
        assert h5BeamCenterX.attrs['units'] == b'pixel'
        h5BeamCenterY = h5DetectorGroup['beam_center_y']
        assert h5BeamCenterY.attrs['units'] == b'pixel'
        h5BitDepthImage = h5DetectorGroup['bit_depth_image']
        h5XPixelSize = h5DetectorGroup['x_pixel_size']
        assert h5XPixelSize.attrs['units'] == b'm'
        h5YPixelSize = h5DetectorGroup['y_pixel_size']
        assert h5YPixelSize.attrs['units'] == b'm'

        detector = DetectorGroup(detectorSpecific=detectorSpecific,
                                 detector_distance_m=h5DetectorDistance[()],
                                 beam_center_x_px=h5BeamCenterX[()],
                                 beam_center_y_px=h5BeamCenterY[()],
                                 bit_depth_image=h5BitDepthImage[()],
                                 x_pixel_size_m=h5XPixelSize[()],
                                 y_pixel_size_m=h5YPixelSize[()])

        return InstrumentGroup(detector=detector)

    @staticmethod
    def _readSampleGroup(h5SampleGroup: Optional[h5py.Group]) -> Optional[SampleGroup]:
        if h5SampleGroup is None:
            return None

        h5GoniometerGroup = h5SampleGroup['goniometer']

        h5ChiDataset = h5GoniometerGroup['chi']
        assert h5ChiDataset.attrs['units'] == b'degree'

        goniometer = GoniometerGroup(chi_deg=float(h5ChiDataset[0]))
        return SampleGroup(goniometer=goniometer)

    def readRootGroup(self, rootGroup: h5py.Group) -> None:
        self.masterFilePath = Path(rootGroup.filename)
        h5EntryGroup = rootGroup.get('entry')

        if h5EntryGroup:
            dataGroup = self._readDataGroup(h5EntryGroup.get('data'))
            instrumentGroup = self._readInstrumentGroup(h5EntryGroup.get('instrument'))
            sampleGroup = self._readSampleGroup(h5EntryGroup.get('sample'))
            self.entryGroup = EntryGroup(data=dataGroup,
                                         instrument=instrumentGroup,
                                         sample=sampleGroup)
        else:
            logger.debug(f'File {self.masterFilePath} is not a velociprobe data file.')
            self.entryGroup = None


class VelociprobeScanYPositionSource(IntEnum):
    LASER_INTERFEROMETER = 2
    ENCODER = 5


class VelociprobeScanPointList:
    def __init__(self) -> None:
        self.xInNanometers: list[int] = list()
        self.yInNanometers: list[int] = list()

    def append(self, x_nm: int, y_nm: int) -> None:
        self.xInNanometers.append(x_nm)
        self.yInNanometers.append(y_nm)

    def mean(self) -> ScanPoint:
        nanometersToMeters = Decimal('1e-9')
        x_nm = Decimal(sum(self.xInNanometers)) / Decimal(len(self.xInNanometers))
        y_nm = Decimal(sum(self.yInNanometers)) / Decimal(len(self.yInNanometers))
        return ScanPoint(x_nm * nanometersToMeters, y_nm * nanometersToMeters)


class VelociprobeScanFileReader(ScanFileReader):
    X_COLUMN = 1
    TRIGGER_COLUMN = 7

    def __init__(self, dataFileReader: VelociprobeDataFileReader,
                 yPositionSource: VelociprobeScanYPositionSource) -> None:
        self._dataFileReader = dataFileReader
        self._yPositionSource = yPositionSource

    @property
    def simpleName(self) -> str:
        yPositionSourceText = 'EncoderY'

        if self._yPositionSource == VelociprobeScanYPositionSource.LASER_INTERFEROMETER:
            yPositionSourceText = 'LaserInterferometerY'

        return f'VelociprobeWith{yPositionSourceText}'

    @property
    def fileFilter(self) -> str:
        yPositionSourceText = 'Encoder Y'

        if self._yPositionSource == VelociprobeScanYPositionSource.LASER_INTERFEROMETER:
            yPositionSourceText = 'Laser Interferometer Y'

        return f'Velociprobe Scan Files - {yPositionSourceText} (*.txt)'

    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        scanPointDict: dict[int, VelociprobeScanPointList] = defaultdict(VelociprobeScanPointList)

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) != 8:
                    raise ScanPointParseError()

                trigger = int(row[VelociprobeScanFileReader.TRIGGER_COLUMN])
                x_nm = int(row[VelociprobeScanFileReader.X_COLUMN])
                y_nm = int(row[self._yPositionSource.value])

                if self._yPositionSource == VelociprobeScanYPositionSource.ENCODER:
                    y_nm = -y_nm

                scanPointDict[trigger].append(x_nm, y_nm)

        scanPointList = [
            scanPointList.mean() for _, scanPointList in sorted(scanPointDict.items())
        ]
        xMeanInMeters = Decimal(sum(point.x for point in scanPointList)) / len(scanPointList)
        yMeanInMeters = Decimal(sum(point.y for point in scanPointList)) / len(scanPointList)

        for idx, scanPoint in enumerate(scanPointList):
            chi_rad = 0.

            if self._dataFileReader.entryGroup and self._dataFileReader.entryGroup.sample and self._dataFileReader.entryGroup.sample.goniometer:
                chi_rad = self._dataFileReader.entryGroup.sample.goniometer.chi_rad

            x_m = (scanPoint.x - xMeanInMeters) * Decimal(numpy.cos(chi_rad))
            y_m = (scanPoint.y - yMeanInMeters)
            scanPointList[idx] = ScanPoint(x_m, y_m)

        return scanPointList
