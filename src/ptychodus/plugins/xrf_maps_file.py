from pathlib import Path
from typing import Final

import numpy

from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)
from ptychodus.api.io import load_fluorescence_data, save_fluorescence_data
from ptychodus.api.plugins import PluginRegistry


class XRFMapsFileIO(FluorescenceFileReader, FluorescenceFileWriter):
    SIMPLE_NAME: Final[str] = 'XRF-Maps'
    DISPLAY_NAME: Final[str] = 'XRF-Maps Fluorescence Dataset (*.h5 *.h5*)'

    def read(self, file_path: Path) -> FluorescenceDataset:
        return load_fluorescence_data(file_path)

    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        save_fluorescence_data(file_path, dataset)


class NPZFluorescenceFileWriter(FluorescenceFileWriter):
    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        contents = {emap.name: emap.counts_per_second for emap in dataset}
        numpy.savez_compressed(file_path, allow_pickle=False, **contents)


def register_plugins(registry: PluginRegistry) -> None:
    xrf_maps_file_io = XRFMapsFileIO()

    registry.fluorescence_file_readers.register_plugin(
        xrf_maps_file_io,
        simple_name=XRFMapsFileIO.SIMPLE_NAME,
        display_name=XRFMapsFileIO.DISPLAY_NAME,
    )
    registry.fluorescence_file_writers.register_plugin(
        xrf_maps_file_io,
        simple_name=XRFMapsFileIO.SIMPLE_NAME,
        display_name=XRFMapsFileIO.DISPLAY_NAME,
    )
    registry.fluorescence_file_writers.register_plugin(
        NPZFluorescenceFileWriter(),
        simple_name='NPZ',
        display_name='NumPy Zipped Archive (*.npz)',
    )
