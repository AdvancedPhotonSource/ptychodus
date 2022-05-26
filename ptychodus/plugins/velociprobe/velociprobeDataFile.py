from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import concurrent.futures
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
        self._dataset: Optional[DiffractionDataset] = None

    @property
    def datasetName(self) -> str:
        return self._name

    @property
    def datasetState(self) -> DatasetState:
        return self._state

    def __getitem__(self, index: int) -> DataArrayType:
        if self._dataset is None:
            self.reloadDataset()

        return numpy.empty((0, 0, 0)) if self._dataset is None else self._dataset[index, ...]

    def __len__(self) -> int:
        if self._dataset is None:
            self.reloadDataset()

        return 0 if self._dataset is None else self._dataset.shape[0]

    def getArray(self) -> DataArrayType:
        if self._dataset is None:
            self.reloadDataset()

        return self._dataset

    def reloadDataset(self) -> VelociprobeDiffractionDataset:
        if self._filePath.is_file():
            self._dataset = None
            self._state = DatasetState.EXISTS

            with h5py.File(self._filePath, 'r') as h5File:
                item = h5File.get(self._dataPath)

                if isinstance(item, h5py.Dataset):
                    self._dataset = item[()]
                    self._state = DatasetState.VALID
                else:
                    logger.error(f'Symlink {self.filePath}:{self.dataPath} is not a dataset!')
        else:
            self._dataset = None
            self._state = DatasetState.NOT_FOUND

        return self


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
        self.entryGroup: Optional[EntryGroup] = None

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return 'Velociprobe Master Files (*.h5 *.hdf5)'

    def _readDataGroup(self, h5DataGroup: h5py.Group, filePath: Path) -> DataGroup:
        if h5DataGroup is None:
            return None

        datasetList = list()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futureList = list()

            for name, h5Item in h5DataGroup.items():
                h5Item = h5DataGroup.get(name, getlink=True)

                if isinstance(h5Item, h5py.ExternalLink):
                    dataset = VelociprobeDiffractionDataset(name=name,
                                                            filePath=filePath.parent /
                                                            h5Item.filename,
                                                            dataPath=str(h5Item.path))
                    future = executor.submit(dataset.reloadDataset)
                    futureList.append(future)

            for future in concurrent.futures.as_completed(futureList):
                try:
                    dataset = future.result()
                except OSError as err:
                    logger.error(err)
                    continue

                if dataset is not None:
                    logger.debug(f'Read {dataset.datasetName}')
                    datasetList.append(dataset)

        datasetList.sort(key=lambda x: x.datasetName)

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

        if filePath is not None:
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
                        dataGroup = self._readDataGroup(h5DataGroup, filePath)
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

        return H5DataFile(filePath, contentsTree, datasetList)
