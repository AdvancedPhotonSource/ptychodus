from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    BadPixels,
    CropCenter,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.tree import SimpleTreeNode

from ..h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataGroup:
    array_list: list[DiffractionArray] = field(default_factory=list)

    @classmethod
    def read(cls, group: h5py.Group, num_patterns_per_array: int) -> DataGroup:
        array_list: list[DiffractionArray] = list()
        master_file_path = Path(group.file.filename)

        for name, h5_item in sorted(group.items()):
            h5_item = group.get(name, getlink=True)

            if isinstance(h5_item, h5py.ExternalLink):
                array = H5DiffractionPatternArray(
                    label=name,
                    indexes=numpy.arange(num_patterns_per_array)
                    + len(array_list) * num_patterns_per_array,
                    file_path=master_file_path.parent / h5_item.filename,
                    data_path=str(h5_item.path),
                )
                array_list.append(array)

        return cls(array_list)

    def __iter__(self) -> Iterator[DiffractionArray]:
        return iter(self.array_list)

    @overload
    def __getitem__(self, index: int) -> DiffractionArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]: ...

    def __getitem__(self, index: int | slice) -> DiffractionArray | Sequence[DiffractionArray]:
        return self.array_list[index]

    def __len__(self) -> int:
        return len(self.array_list)


@dataclass(frozen=True)
class DetectorSpecificGroup:
    nimages: int
    ntrigger: int
    photon_energy_eV: float  # noqa: N815
    x_pixels_in_detector: int
    y_pixels_in_detector: int

    @property
    def num_patterns_total(self) -> int:
        return max(self.nimages, self.ntrigger)

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorSpecificGroup:
        nimages = group['nimages']
        ntrigger = group['ntrigger']
        photon_energy = group['photon_energy']
        assert photon_energy.attrs['units'] == b'eV'
        x_pixels_in_detector = group['x_pixels_in_detector']
        y_pixels_in_detector = group['y_pixels_in_detector']
        return cls(
            int(nimages[()]),
            int(ntrigger[()]),
            float(photon_energy[()]),
            int(x_pixels_in_detector[()]),
            int(y_pixels_in_detector[()]),
        )


@dataclass(frozen=True)
class DetectorGroup:
    detector_specific: DetectorSpecificGroup
    detector_distance_m: float
    beam_center_x_px: int
    beam_center_y_px: int
    bit_depth_readout: int
    x_pixel_size_m: float
    y_pixel_size_m: float

    @classmethod
    def read(cls, group: h5py.Group) -> DetectorGroup:
        detector_specific = DetectorSpecificGroup.read(group['detectorSpecific'])
        h5_detector_distance = group['detector_distance']
        assert h5_detector_distance.attrs['units'] == b'm'
        h5_beam_center_x = group['beam_center_x']
        assert h5_beam_center_x.attrs['units'] == b'pixel'
        h5_beam_center_y = group['beam_center_y']
        assert h5_beam_center_y.attrs['units'] == b'pixel'
        h5_bit_depth_readout = group['bit_depth_readout']
        h5_x_pixel_size = group['x_pixel_size']
        assert h5_x_pixel_size.attrs['units'] == b'm'
        h5_y_pixel_size = group['y_pixel_size']
        assert h5_y_pixel_size.attrs['units'] == b'm'
        return cls(
            detector_specific,
            float(h5_detector_distance[()]),
            int(h5_beam_center_x[()]),
            int(h5_beam_center_y[()]),
            int(h5_bit_depth_readout[()]),
            float(h5_x_pixel_size[()]),
            float(h5_y_pixel_size[()]),
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
    chi_deg: float

    @classmethod
    def read(cls, group: h5py.Group) -> GoniometerGroup:
        chi_item = group['chi']
        chi_space = chi_item.id.get_space()

        assert chi_item.attrs['units'] == b'degree'

        if chi_space.get_simple_extent_type() == h5py.h5s.SCALAR:
            chi_deg = float(chi_item[()])
        elif isinstance(chi_item, h5py.Dataset):
            chi_deg = float(chi_item[0])
        else:
            raise ValueError('Failed to read goniometer angle (chi)!')

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
    def read(cls, group: h5py.Group, num_patterns_per_array: int) -> EntryGroup:
        data = DataGroup.read(group['data'], num_patterns_per_array)
        instrument = InstrumentGroup.read(group['instrument'])
        sample = SampleGroup.read(group['sample'])
        return cls(data, instrument, sample)


class VelociprobeDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        metadata: DiffractionMetadata,
        contents_tree: SimpleTreeNode,
        entry: EntryGroup,
    ) -> None:
        self._metadata = metadata
        self._contents_tree = contents_tree
        self._entry = entry

    def get_metadata(self) -> DiffractionMetadata:
        return self._metadata

    def get_layout(self) -> SimpleTreeNode:
        return self._contents_tree

    def get_bad_pixels(self) -> BadPixels | None:
        return None

    @overload
    def __getitem__(self, index: int) -> DiffractionArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]: ...

    def __getitem__(self, index: int | slice) -> DiffractionArray | Sequence[DiffractionArray]:
        return self._entry.data[index]

    def __len__(self) -> int:
        return len(self._entry.data)


class VelociprobeDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        super().__init__()
        self._tree_builder = H5DiffractionFileTreeBuilder()
        self.stage_rotation_deg = 0.0  # TODO This is a hack; remove when able!

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset: DiffractionDataset = SimpleDiffractionDataset.create_null(file_path)

        with h5py.File(file_path, 'r') as h5_file:
            h5_dataset = h5_file['/entry/data/data_000001']
            num_patterns_per_array = h5_dataset.shape[0]
            pattern_dtype = h5_dataset.dtype

            entry = EntryGroup.read(h5_file['entry'], num_patterns_per_array)
            detector = entry.instrument.detector
            detector_pixel_geometry = PixelGeometry(
                detector.x_pixel_size_m,
                detector.y_pixel_size_m,
            )
            crop_center = CropCenter(
                detector.beam_center_x_px,
                detector.beam_center_y_px,
            )
            detector_specific = detector.detector_specific
            detector_extent = ImageExtent(
                detector_specific.x_pixels_in_detector,
                detector_specific.y_pixels_in_detector,
            )
            probe_energy_eV = detector_specific.photon_energy_eV  # noqa: N806
            num_arrays = detector_specific.num_patterns_total // num_patterns_per_array

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns_per_array] * num_arrays,
                pattern_dtype=pattern_dtype,
                detector_distance_m=detector.detector_distance_m,
                detector_extent=detector_extent,
                detector_pixel_geometry=detector_pixel_geometry,
                detector_bit_depth=detector.bit_depth_readout,
                crop_center=crop_center,
                probe_energy_eV=probe_energy_eV,
                file_path=file_path,
            )
            contents_tree = self._tree_builder.build(h5_file)
            dataset = VelociprobeDiffractionDataset(metadata, contents_tree, entry)

            # vvv TODO This is a hack; remove when able! vvv
            self.stage_rotation_deg = entry.sample.goniometer.chi_deg

        return dataset
