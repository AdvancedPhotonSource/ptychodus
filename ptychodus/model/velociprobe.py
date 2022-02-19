from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

import h5py

from .data_file import DataFileReader
from .observer import Observable, Observer


class DatasetState(Enum):
    MISSING = auto()
    FOUND = auto()
    VALID = auto()


@dataclass(frozen=True)
class Datafile:
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
    datafileList: list[Datafile] = field(default_factory=list)

    def __iter__(self):
        return iter(self.datafileList)

    def __getitem__(self, index: int) -> Datafile:
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
        self.entryGroup = None

    def _readDataGroup(self, h5DataGroup: h5py.Group, masterFilePath: Path) -> DataGroup:
        datafileList = list()

        for name, h5Item in h5DataGroup.items():
            h5Item = h5DataGroup.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink):
                datafile = Datafile(name=name,
                                    filePath=masterFilePath.parent / h5Item.filename,
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
        masterFilePath = Path(rootGroup.filename)

        h5EntryGroup = rootGroup['entry']
        self.entryGroup = EntryGroup(
            data=self._readDataGroup(h5EntryGroup['data'], masterFilePath),
            instrument=self._readInstrumentGroup(h5EntryGroup['instrument']),
            sample=self._readSampleGroup(h5EntryGroup['sample']))
        self.notifyObservers()
