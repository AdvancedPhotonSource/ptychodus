from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    CropCenter,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatternArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.tree import SimpleTreeNode

from ..h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataGroup:
    arrayList: list[DiffractionPatternArray] = field(default_factory=list)

    @classmethod
    def read(cls, group: h5py.Group, numberOfPatternsPerArray: int) -> DataGroup:
        arrayList: list[DiffractionPatternArray] = list()
        masterFilePath = Path(group.file.filename)

        for name, h5Item in sorted(group.items()):
            h5Item = group.get(name, getlink=True)

            if isinstance(h5Item, h5py.ExternalLink):
                array = H5DiffractionPatternArray(
                    label=name,
                    indexes=numpy.arange(numberOfPatternsPerArray)
                    + len(arrayList) * numberOfPatternsPerArray,
                    filePath=masterFilePath.parent / h5Item.filename,
                    dataPath=str(h5Item.path),
                )
                arrayList.append(array)

        return cls(arrayList)

    def __iter__(self) -> Iterator[DiffractionPatternArray]:
        return iter(self.arrayList)

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
        return self.arrayList[index]

    def __len__(self) -> int:
        return len(self.arrayList)


@dataclass(frozen=True)
class DetectorSpecificGroup:
    nimages: int
    ntrigger: int
    photonEnergyInElectronVolts: float
    xPixelsInDetector: int
    yPixelsInDetector: int

    @property
    def numberOfPatternsTotal(self) -> int:
        return max(self.nimages, self.ntrigger)

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorSpecificGroup:
        nimages = group['nimages']
        ntrigger = group['ntrigger']
        photonEnergy = group['photon_energy']
        assert photonEnergy.attrs['units'] == b'eV'
        xPixelsInDetector = group['x_pixels_in_detector']
        yPixelsInDetector = group['y_pixels_in_detector']
        return cls(
            int(nimages[()]),
            int(ntrigger[()]),
            float(photonEnergy[()]),
            int(xPixelsInDetector[()]),
            int(yPixelsInDetector[()]),
        )


@dataclass(frozen=True)
class DetectorGroup:
    detectorSpecific: DetectorSpecificGroup
    detectorDistanceInMeters: float
    beamCenterXInPixels: int
    beamCenterYInPixels: int
    bitDepthReadout: int
    xPixelSizeInMeters: float
    yPixelSizeInMeters: float

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorGroup:
        detectorSpecific = DetectorSpecificGroup.read(group['detectorSpecific'])
        h5DetectorDistance = group['detector_distance']
        assert h5DetectorDistance.attrs['units'] == b'm'
        h5BeamCenterX = group['beam_center_x']
        assert h5BeamCenterX.attrs['units'] == b'pixel'
        h5BeamCenterY = group['beam_center_y']
        assert h5BeamCenterY.attrs['units'] == b'pixel'
        h5BitDepthReadout = group['bit_depth_readout']
        h5XPixelSize = group['x_pixel_size']
        assert h5XPixelSize.attrs['units'] == b'm'
        h5YPixelSize = group['y_pixel_size']
        assert h5YPixelSize.attrs['units'] == b'm'
        return cls(
            detectorSpecific,
            float(h5DetectorDistance[()]),
            int(h5BeamCenterX[()]),
            int(h5BeamCenterY[()]),
            int(h5BitDepthReadout[()]),
            float(h5XPixelSize[()]),
            float(h5YPixelSize[()]),
        )


@dataclass(frozen=True)
class InstrumentGroup:
    detector: DetectorGroup

    @classmethod
    def read(cls, group: h5py.Group) -> InstrumentGroup:
        detector = DetectorGroup.read(group['detector'])
        return cls(detector)


@dataclass(frozen=True)
class GoniometerGroup:
    chiDeg: float

    @classmethod
    def read(cls, group: h5py.Group) -> GoniometerGroup:
        chiItem = group['chi']
        chiSpace = chiItem.id.get_space()

        assert chiItem.attrs['units'] == b'degree'

        if chiSpace.get_simple_extent_type() == h5py.h5s.SCALAR:
            chiDeg = float(chiItem[()])
        elif isinstance(chiItem, h5py.Dataset):
            chiDeg = float(chiItem[0])
        else:
            raise ValueError('Failed to read goniometer angle (chi)!')

        return cls(chiDeg)


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
    def read(cls, group: h5py.Group, numberOfPatternsPerArray: int) -> EntryGroup:
        data = DataGroup.read(group['data'], numberOfPatternsPerArray)
        instrument = InstrumentGroup.read(group['instrument'])
        sample = SampleGroup.read(group['sample'])
        return cls(data, instrument, sample)


class NeXusDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        metadata: DiffractionMetadata,
        contentsTree: SimpleTreeNode,
        entry: EntryGroup,
    ) -> None:
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._entry = entry

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
        return self._entry.data[index]

    def __len__(self) -> int:
        return len(self._entry.data)


class NeXusDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        super().__init__()
        self._treeBuilder = H5DiffractionFileTreeBuilder()
        self.stageRotationInDegrees = 0.0  # TODO This is a hack; remove when able!

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset: DiffractionDataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                metadata = DiffractionMetadata.createNullInstance(filePath)
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    h5Dataset = h5File['/entry/data/data_000001']
                except KeyError:
                    logger.error(f'File {filePath} is not a NeXus data file.')
                    raise

                numberOfPatternsPerArray = h5Dataset.shape[0]
                patternDataType = h5Dataset.dtype

                try:
                    entry = EntryGroup.read(h5File['entry'], numberOfPatternsPerArray)
                except KeyError:
                    logger.error(f'File {filePath} is not a NeXus data file.')
                    raise

                detector = entry.instrument.detector
                detectorPixelGeometry = PixelGeometry(
                    detector.xPixelSizeInMeters,
                    detector.yPixelSizeInMeters,
                )
                cropCenter = CropCenter(
                    detector.beamCenterXInPixels,
                    detector.beamCenterYInPixels,
                )

                detectorSpecific = detector.detectorSpecific
                detectorExtent = ImageExtent(
                    detectorSpecific.xPixelsInDetector,
                    detectorSpecific.yPixelsInDetector,
                )
                probeEnergyInElectronVolts = detectorSpecific.photonEnergyInElectronVolts

                metadata = DiffractionMetadata(
                    numberOfPatternsPerArray=numberOfPatternsPerArray,
                    numberOfPatternsTotal=detectorSpecific.numberOfPatternsTotal,
                    patternDataType=patternDataType,
                    detectorDistanceInMeters=detector.detectorDistanceInMeters,
                    detectorExtent=detectorExtent,
                    detectorPixelGeometry=detectorPixelGeometry,
                    detectorBitDepth=detector.bitDepthReadout,
                    cropCenter=cropCenter,
                    probeEnergyInElectronVolts=probeEnergyInElectronVolts,
                    filePath=filePath,
                )

                dataset = NeXusDiffractionDataset(metadata, contentsTree, entry)

                # vvv TODO This is a hack; remove when able! vvv
                self.stageRotationInDegrees = entry.sample.goniometer.chiDeg
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset
