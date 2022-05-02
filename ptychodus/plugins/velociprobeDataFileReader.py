from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import logging

import h5py
import numpy

from .h5DataFileReader import H5DataFileReader
from ptychodus.api.plugins import PluginRegistry

logger = logging.getLogger(__name__)


class DatasetState(Enum):
    MISSING = auto()
    FOUND = auto()
    VALID = auto()


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


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(VelociprobeDataFileReader())
