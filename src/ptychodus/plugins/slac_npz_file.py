from pathlib import Path
from typing import Final, Sequence
import logging

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
    SimpleDiffractionArray,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, Product, ProductFileReader, ProductMetadata
from ptychodus.api.scan import PositionSequence, ScanPoint
from ptychodus.api.tree import SimpleTreeNode

logger = logging.getLogger(__name__)


class SLACDiffractionFileReader(DiffractionFileReader):
    def read(self, file_path: Path) -> DiffractionDataset:
        with numpy.load(file_path) as npz_file:
            patterns = numpy.transpose(npz_file['diffraction'], [2, 0, 1])

        num_patterns, detector_height, detector_width = patterns.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=patterns.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            file_path=file_path,
        )

        contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
        contents_tree.create_child(
            [file_path.stem, type(patterns).__name__, f'{patterns.dtype}{patterns.shape}']
        )

        array = SimpleDiffractionArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            patterns=patterns,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array])


class SLACProductFileReader(ProductFileReader):
    def read(self, file_path: Path) -> Product:
        with numpy.load(file_path) as npz_file:
            scan_x_m = npz_file['xcoords_start']
            scan_y_m = npz_file['ycoords_start']
            probe_array = npz_file['probeGuess']
            object_array = npz_file['objectGuess']

        metadata = ProductMetadata(
            name=file_path.stem,
            comments='',
            detector_distance_m=0.0,  # not included in file
            probe_energy_eV=0.0,  # not included in file
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        point_list: list[ScanPoint] = list()

        for idx, (x_m, y_m) in enumerate(zip(scan_x_m, scan_y_m)):
            point = ScanPoint(idx, x_m, y_m)
            point_list.append(point)

        loss: Sequence[LossValue] = list()  # not included in file

        return Product(
            metadata=metadata,
            positions=PositionSequence(point_list),
            probes=ProbeSequence(array=probe_array, opr_weights=None, pixel_geometry=None),
            object_=Object(array=object_array, pixel_geometry=None, center=None),
            losses=loss,
        )


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'SLAC_NPZ'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'SLAC NumPy Zipped Archive (*.npz)'  # noqa: N806

    registry.diffraction_file_readers.register_plugin(
        SLACDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        SLACProductFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
