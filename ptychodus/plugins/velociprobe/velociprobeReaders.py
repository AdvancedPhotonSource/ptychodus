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

from ..h5DataFileReader import H5DataFileTreeBuilder, H5DataFile
from ptychodus.api.data import *
from ptychodus.api.scan import ScanFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataGroup:
    datasetList: list[DiffractionDataset] = field(default_factory=list)

    def __iter__(self):
        return iter(self.datasetList)

    def __getitem__(self, index: int) -> DataFile:
        return self.datasetList[index]

    def __len__(self) -> int:
        return len(self.datasetList)


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
    data: DataGroup
    instrument: InstrumentGroup
    sample: SampleGroup


class VelociprobeDataFileReader(DataFileReader, Observable):
    def __init__(self) -> None:
        super().__init__()
        self._treeBuilder = H5DataFileTreeBuilder()
        self.masterFilePath: Optional[Path] = None
        self.entryGroup: Optional[EntryGroup] = None

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return 'Velociprobe Master Files (*.h5 *.hdf5)'

    def _readDataGroup(self, h5DataGroup: h5py.Group) -> DataGroup:
        if h5DataGroup is None:
            return None

        datasetList = list()

        for name, h5Item in h5DataGroup.items():
            h5Item = h5DataGroup.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink) and self.masterFilePath is not None:
                dataset = DataFile(name=name,
                                    filePath=self.masterFilePath.parent / h5Item.filename,
                                    dataPath=str(h5Item.path))
                datasetList.append(dataset)

        datasetList.sort(key=lambda x: x.name)

        return DataGroup(datasetList)

    @staticmethod
    def _readInstrumentGroup(h5InstrumentGroup: h5py.Group) -> InstrumentGroup:
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
    def _readSampleGroup(h5SampleGroup: h5py.Group) -> SampleGroup:
        if h5SampleGroup is None:
            return None

        h5GoniometerGroup = h5SampleGroup['goniometer']

        h5ChiDataset = h5GoniometerGroup['chi']
        assert h5ChiDataset.attrs['units'] == b'degree'

        goniometer = GoniometerGroup(chi_deg=float(h5ChiDataset[0]))
        return SampleGroup(goniometer=goniometer)

    def read(self, filePath: Path) -> DataFile:
        contentsTree = self._treeBuilder.createRootNode()
        datasetList: list[DiffractionDataset] = list()

        if filePath:
            self.masterFilePath = filePath

            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)
                h5EntryGroup = h5File.get('entry')

                if h5EntryGroup:
                    try:
                        h5DataGroup = h5EntryGroup['data']
                        h5InstrumentGroup = h5EntryGroup['instrument']
                        h5SampleGroup = h5EntryGroup['sample']
                    except KeyError:
                        logger.info(f'File {filePath} is not a velociprobe data file.')
                        self.entryGroup = None
                    else:
                        dataGroup = self._readDataGroup(h5DataGroup)
                        instrumentGroup = self._readInstrumentGroup(h5InstrumentGroup)
                        sampleGroup = self._readSampleGroup(h5SampleGroup)

                        self.entryGroup = EntryGroup(data=dataGroup,
                                                     instrument=instrumentGroup,
                                                     sample=sampleGroup)
                        datasetList = list(dataGroup.datasetList)
                else:
                    logger.info(f'File {filePath} is not a velociprobe data file.')
                    self.entryGroup = None

            self.notifyObservers()

        return H5DataFile(contentsTree, datasetList)


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

            if self._dataFileReader.entryGroup:
                chi_rad = self._dataFileReader.entryGroup.sample.goniometer.chi_rad

            x_m = (scanPoint.x - xMeanInMeters) * Decimal(numpy.cos(chi_rad))
            y_m = (scanPoint.y - yMeanInMeters)
            scanPointList[idx] = ScanPoint(x_m, y_m)

        return scanPointList
