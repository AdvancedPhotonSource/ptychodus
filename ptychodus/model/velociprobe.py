from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional, Tuple
import csv
import logging

import h5py
import numpy

from .crop import CropSettings
from .data import DataFile, DataFileReader
from .detector import DetectorSettings
from .image import ImageSequence
from .observer import Observable, Observer
from .probe import ProbeSettings
from .scan import ScanFileReader, ScanPoint, ScanPointParseError
from .settings import SettingsGroup

logger = logging.getLogger(__name__)


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


class VelociprobeReader(DataFileReader, Observable):
    def __init__(self) -> None:
        super().__init__()
        self.masterFilePath: Optional[Path] = None
        self.entryGroup: Optional[EntryGroup] = None

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return f'Velociprobe Master Files (*.h5)'

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

    def read(self, filePath: Path) -> None:
        if not filePath:
            return

        with h5py.File(filePath, 'r') as h5File:
            self.masterFilePath = Path(h5File.filename)
            h5EntryGroup = h5File.get('entry')

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

            self.notifyObservers()


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

    def __init__(self, velociprobeReader: VelociprobeReader,
                 yPositionSource: VelociprobeScanYPositionSource) -> None:
        self._velociprobeReader = velociprobeReader
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

            if self._velociprobeReader.entryGroup and self._velociprobeReader.entryGroup.sample and self._velociprobeReader.entryGroup.sample.goniometer:
                chi_rad = self._velociprobeReader.entryGroup.sample.goniometer.chi_rad

            x_m = (scanPoint.x - xMeanInMeters) * Decimal(numpy.cos(chi_rad))
            y_m = (scanPoint.y - yMeanInMeters)
            scanPointList[idx] = ScanPoint(x_m, y_m)

        return scanPointList


class VelociprobeImageSequence(ImageSequence):
    def __init__(self, velociprobeReader: VelociprobeReader) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader
        self._datasetImageList: list[numpy.ndarray] = list()
        self._datasetIndex = -1

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader) -> VelociprobeImageSequence:
        imageSequence = cls(velociprobeReader)
        imageSequence._updateImages()
        velociprobeReader.addObserver(imageSequence)
        return imageSequence

    def setCurrentDatasetIndex(self, index: int) -> None:
        if index < 0:
            raise IndexError('Current dataset index must be non-negative.')

        self._datasetIndex = index
        self._updateImages()

    def getCurrentDatasetIndex(self) -> int:
        return self._datasetIndex

    def getWidth(self) -> int:
        value = 0

        if self._datasetImageList:
            value = self._datasetImageList[0].shape[1]

        return value

    def getHeight(self) -> int:
        value = 0

        if self._datasetImageList:
            value = self._datasetImageList[0].shape[0]

        return value

    def __getitem__(self, index: int) -> numpy.ndarray:
        return self._datasetImageList[index]

    def __len__(self) -> int:
        return len(self._datasetImageList)

    def _updateImages(self) -> None:
        self._datasetImageList = list()

        if self._velociprobeReader.entryGroup and self._velociprobeReader.entryGroup.data:
            if self._datasetIndex >= len(self._velociprobeReader.entryGroup.data):
                self._datasetIndex = 0
        else:
            self._datasetIndex = 0
            return

        datafile = self._velociprobeReader.entryGroup.data[self._datasetIndex]

        if datafile.filePath.is_file():
            logger.debug(f'Reading {datafile.filePath}/{datafile.dataPath}')

            with h5py.File(datafile.filePath, 'r') as h5File:
                item = h5File.get(datafile.dataPath)

                if isinstance(item, h5py.Dataset):
                    data = item[()]

                    for imslice in data:
                        image = numpy.copy(imslice)
                        self._datasetImageList.append(image)
                else:
                    logger.error('Data path does not refer to a dataset.')
        else:
            logger.error(f'File {datafile.filePath} not found.')

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self._updateImages()


class VelociprobePresenter(Observable, Observer):
    def __init__(self, velociprobeReader: VelociprobeReader, detectorSettings: DetectorSettings,
                 cropSettings: CropSettings, probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings
        self._probeSettings = probeSettings

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader,
                       detectorSettings: DetectorSettings, cropSettings: CropSettings,
                       probeSettings: ProbeSettings):
        presenter = cls(velociprobeReader, detectorSettings, cropSettings, probeSettings)
        velociprobeReader.addObserver(presenter)
        return presenter

    def getDatasetName(self, index: int) -> str:
        datafile = self._velociprobeReader.entryGroup.data[index]
        return datafile.name

    def getDatasetState(self, index: int) -> DatasetState:
        datafile = self._velociprobeReader.entryGroup.data[index]
        return datafile.getState()

    def getNumberOfDatasets(self) -> int:
        return 0 if self._velociprobeReader.entryGroup is None \
                else len(self._velociprobeReader.entryGroup.data)

    @property
    def _detectorGroup(self) -> DetectorGroup:
        return self._velociprobeReader.entryGroup.instrument.detector

    @property
    def _detectorSpecificGroup(self) -> DetectorSpecificGroup:
        return self._detectorGroup.detectorSpecific

    def syncDetectorPixelCount(self) -> None:
        self._detectorSettings.numberOfPixelsX.value = \
                int(self._detectorSpecificGroup.x_pixels_in_detector)
        self._detectorSettings.numberOfPixelsY.value = \
                int(self._detectorSpecificGroup.y_pixels_in_detector)

    def syncDetectorPixelSize(self) -> None:
        self._detectorSettings.pixelSizeXInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.x_pixel_size_m)
        self._detectorSettings.pixelSizeYInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.y_pixel_size_m)

    def syncDetectorDistance(self, overrideDistanceUnits: bool = False) -> None:
        value = SettingsGroup.convertFloatToDecimal(self._detectorGroup.detector_distance_m)

        if overrideDistanceUnits:
            value /= 1000

        self._detectorSettings.detectorDistanceInMeters.value = value

    def syncImageCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            self._cropSettings.centerXInPixels.value = \
                    int(round(self._detectorGroup.beam_center_x_px))
            self._cropSettings.centerYInPixels.value = \
                    int(round(self._detectorGroup.beam_center_y_px))

        if syncExtent:
            centerX = self._cropSettings.centerXInPixels.value
            centerY = self._cropSettings.centerYInPixels.value

            extentX = int(self._detectorSpecificGroup.x_pixels_in_detector)
            extentY = int(self._detectorSpecificGroup.y_pixels_in_detector)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._cropSettings.extentXInPixels.value = cropDiameterInPixels
            self._cropSettings.extentYInPixels.value = cropDiameterInPixels

    def syncProbeEnergy(self) -> None:
        self._probeSettings.probeEnergyInElectronVolts.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorSpecificGroup.photon_energy_eV)

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self.notifyObservers()
