from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy

from .data_file import DataFile, DataFileReader
from .image import ImageSequence
from .observer import Observable, Observer


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
class ModuleGroup:
    x_data_size: int
    y_data_size: int


@dataclass(frozen=True)
class DetectorGroup:
    detectorSpecific: DetectorSpecificGroup
    module: ModuleGroup
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


@dataclass(frozen=True)
class SampleGroup:
    goniometer: GoniometerGroup


@dataclass(frozen=True)
class EntryGroup:
    data: DataGroup
    instrument: InstrumentGroup
    sample: SampleGroup


class VelociprobeReader(DataFileReader, Observable):
    def __init__(self) -> None:
        super().__init__()
        self.masterFilePath: Optional[Path] = None
        self.entryGroup: Optional[EntryGroup] = None

    def _readDataGroup(self, h5DataGroup: h5py.Group) -> DataGroup:
        datafileList = list()

        for name, h5Item in h5DataGroup.items():
            h5Item = h5DataGroup.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink):
                datafile = DataFile(name=name,
                                    filePath=self.masterFilePath.parent / h5Item.filename,
                                    dataPath=str(h5Item.path))
                datafileList.append(datafile)

        datafileList.sort(key=lambda x: x.name)

        return DataGroup(datafileList)

    @staticmethod
    def _readInstrumentGroup(h5InstrumentGroup: h5py.Group) -> InstrumentGroup:
        h5DetectorGroup = h5InstrumentGroup['detector']
        h5DetectorSpecificGroup = h5DetectorGroup['detectorSpecific']
        h5ModuleGroup = h5DetectorGroup['module']

        h5PhotonEnergy = h5DetectorSpecificGroup['photon_energy']
        assert h5PhotonEnergy.attrs['units'] == b'eV'
        h5XPixelsInDetector = h5DetectorSpecificGroup['x_pixels_in_detector']
        h5YPixelsInDetector = h5DetectorSpecificGroup['y_pixels_in_detector']

        detectorSpecific = DetectorSpecificGroup(photon_energy_eV=h5PhotonEnergy[()],
                                                 x_pixels_in_detector=h5XPixelsInDetector[()],
                                                 y_pixels_in_detector=h5YPixelsInDetector[()])

        h5DataSize = h5ModuleGroup['data_size']
        assert len(h5DataSize) == 2

        module = ModuleGroup(x_data_size=h5DataSize[0], y_data_size=h5DataSize[1])

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
                                 module=module,
                                 detector_distance_m=h5DetectorDistance[()],
                                 beam_center_x_px=h5BeamCenterX[()],
                                 beam_center_y_px=h5BeamCenterY[()],
                                 bit_depth_image=h5BitDepthImage[()],
                                 x_pixel_size_m=h5XPixelSize[()],
                                 y_pixel_size_m=h5YPixelSize[()])

        return InstrumentGroup(detector=detector)

    @staticmethod
    def _readSampleGroup(h5SampleGroup: h5py.Group) -> SampleGroup:
        h5GoniometerGroup = h5SampleGroup['goniometer']

        h5ChiDataset = h5GoniometerGroup['chi']
        assert h5ChiDataset.attrs['units'] == b'degree'

        goniometer = GoniometerGroup(chi_deg=h5ChiDataset[0])
        return SampleGroup(goniometer=goniometer)

    def read(self, rootGroup: h5py.Group) -> None:
        self.masterFilePath = Path(rootGroup.filename)

        h5EntryGroup = rootGroup['entry']
        self.entryGroup = EntryGroup(
            data=self._readDataGroup(h5EntryGroup['data']),
            instrument=self._readInstrumentGroup(h5EntryGroup['instrument']),
            sample=self._readSampleGroup(h5EntryGroup['sample']))
        self.notifyObservers()


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

        if self._velociprobeReader.entryGroup is None:
            self._datasetIndex = 0
            return
        elif self._datasetIndex >= len(self._velociprobeReader.entryGroup.data):
            self._datasetIndex = 0

        datafile = self._velociprobeReader.entryGroup.data[self._datasetIndex]

        with h5py.File(datafile.filePath, 'r') as h5File:
            item = h5File.get(datafile.dataPath)

            if isinstance(item, h5py.Dataset):
                data = item[()]

                for imslice in data:
                    image = numpy.copy(imslice)
                    self._datasetImageList.append(image)
            else:
                raise TypeError('Data path does not refer to a dataset.')

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self._updateImages()
