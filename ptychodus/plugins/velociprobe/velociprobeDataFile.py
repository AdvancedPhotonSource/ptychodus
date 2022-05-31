from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import h5py
import numpy

from ..h5DataFile import H5DataFileTreeBuilder, H5DataFile
from ptychodus.api.data import *

logger = logging.getLogger(__name__)


class VelociprobeDiffractionDataset(DiffractionDataset):

    def __init__(self, name: str, filePath: Path, dataPath: str) -> None:
        super().__init__()
        self._name = name
        self._state = DatasetState.NOT_FOUND
        self._filePath = filePath
        self._dataPath = dataPath

    @property
    def datasetName(self) -> str:
        return self._name

    @property
    def datasetState(self) -> DatasetState:
        return self._state

    def __getitem__(self, index: int) -> DataArrayType:
        array = self.getArray()
        return array[index, ...]

    def __len__(self) -> int:
        array = self.getArray()
        return array.shape[0]

    def getArray(self) -> DataArrayType:
        array = numpy.empty((0, 0, 0), dtype=int)

        if self._filePath.is_file():
            self._state = DatasetState.EXISTS

            try:
                with h5py.File(self._filePath, 'r') as h5File:
                    item = h5File.get(self._dataPath)

                    if isinstance(item, h5py.Dataset):
                        array = item[()]
                        self._state = DatasetState.VALID
                    else:
                        logger.error(f'Symlink {self.filePath}:{self.dataPath} is not a dataset!')
            except OSError as err:
                logger.exception(err)
        else:
            self._state = DatasetState.NOT_FOUND

        return array


@dataclass(frozen=True)
class DataGroup:
    datasetList: list[DiffractionDataset] = field(default_factory=list)

    @classmethod
    def read(cls, group: h5py.Group) -> DataGroup:
        datasetList = list()
        masterFilePath = Path(group.file.filename)

        for name, h5Item in group.items():
            h5Item = group.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink):
                filePath = masterFilePath.parent / h5Item.filename
                dataPath = str(h5Item.path)
                dataset = VelociprobeDiffractionDataset(name, filePath, dataPath)
                datasetList.append(dataset)

        return cls(datasetList)

    def __iter__(self):
        return iter(self.datasetList)

    def __getitem__(self, index: int) -> DataFile:
        return self.datasetList[index]

    def __len__(self) -> int:
        return len(self.datasetList)


@dataclass(frozen=True)
class DetectorSpecificGroup:
    nimages: int
    photon_energy_eV: float
    x_pixels_in_detector: int
    y_pixels_in_detector: int

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorSpecificGroup:
        nimages = group['nimages']
        photonEnergy = group['photon_energy']
        assert photonEnergy.attrs['units'] == b'eV'
        xPixelsInDetector = group['x_pixels_in_detector']
        yPixelsInDetector = group['y_pixels_in_detector']
        return cls(nimages[()], photonEnergy[()], xPixelsInDetector[()], yPixelsInDetector[()])


@dataclass(frozen=True)
class DetectorGroup:
    detectorSpecific: DetectorSpecificGroup
    detector_distance_m: float
    beam_center_x_px: int
    beam_center_y_px: int
    bit_depth_image: int
    x_pixel_size_m: float
    y_pixel_size_m: float

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorGroup:
        detectorSpecific = DetectorSpecificGroup.read(group['detectorSpecific'])
        h5DetectorDistance = group['detector_distance']
        assert h5DetectorDistance.attrs['units'] == b'm'
        h5BeamCenterX = group['beam_center_x']
        assert h5BeamCenterX.attrs['units'] == b'pixel'
        h5BeamCenterY = group['beam_center_y']
        assert h5BeamCenterY.attrs['units'] == b'pixel'
        h5BitDepthImage = group['bit_depth_image']
        h5XPixelSize = group['x_pixel_size']
        assert h5XPixelSize.attrs['units'] == b'm'
        h5YPixelSize = group['y_pixel_size']
        assert h5YPixelSize.attrs['units'] == b'm'
        return cls(detectorSpecific, h5DetectorDistance[()], h5BeamCenterX[()], h5BeamCenterY[()],
                   h5BitDepthImage[()], h5XPixelSize[()], h5YPixelSize[()])


@dataclass(frozen=True)
class InstrumentGroup:
    detector: DetectorGroup

    @classmethod
    def read(cls, group: h5py.Group) -> InstrumentGroup:
        detector = DetectorGroup.read(group['detector'])
        return cls(detector)


@dataclass(frozen=True)
class GoniometerGroup:
    chi_deg: float

    @property
    def chi_rad(self) -> float:
        return numpy.deg2rad(self.chi_deg)

    @classmethod
    def read(cls, group: h5py.Group) -> GoniometerGroup:
        chiDataset = group['chi']
        assert chiDataset.attrs['units'] == b'degree'
        chi_deg = float(chiDataset[0])
        return cls(chi_deg)


@dataclass(frozen=True)
class SampleGroup:
    goniometer: GoniometerGroup

    @classmethod
    def read(cls, group: h5py.Group) -> SampleGroup:
        goniometer = GoniometerGroup.read(group['goniometer'])
        return cls(goniometer)


@dataclass(frozen=True)
class EntryGroup:
    data: DataGroup
    instrument: InstrumentGroup
    sample: SampleGroup

    @classmethod
    def read(cls, group: h5py.Group) -> EntryGroup:
        data = DataGroup.read(group['data'])
        instrument = InstrumentGroup.read(group['instrument'])
        sample = SampleGroup.read(group['sample'])
        return cls(data, instrument, sample)


class VelociprobeDataFileReader(DataFileReader, Observable):

    def __init__(self) -> None:
        super().__init__()
        self._treeBuilder = H5DataFileTreeBuilder()
        self.entry: Optional[EntryGroup] = None

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return 'Velociprobe Master Files (*.h5 *.hdf5)'

    def read(self, filePath: Path) -> DataFile:
        metadata = DataFileMetadata(filePath, 0, 0, 0)
        contentsTree = self._treeBuilder.createRootNode()
        datasetList: list[DiffractionDataset] = list()

        if filePath is not None:
            with h5py.File(filePath, 'r') as h5File:
                try:
                    self.entry = EntryGroup.read(h5File['entry'])
                except KeyError:
                    self.entry = None
                    logger.info(f'File {filePath} is not a velociprobe data file.')
                else:
                    detectorSpecific = self.entry.instrument.detector.detectorSpecific
                    metadata = DataFileMetadata(filePath=filePath,
                                                imageWidth=detectorSpecific.x_pixels_in_detector,
                                                imageHeight=detectorSpecific.y_pixels_in_detector,
                                                totalNumberOfImages=detectorSpecific.nimages)
                    contentsTree = self._treeBuilder.build(h5File)
                    datasetList = list(self.entry.data.datasetList)

            self.notifyObservers()

        return H5DataFile(metadata, contentsTree, datasetList)
