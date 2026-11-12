from pathlib import Path
from typing import Final

from ptychodus.api.fluorescence import (
    FluorescenceDataset,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)
from ptychodus.api.io import load_fluorescence_data, save_fluorescence_data
from ptychodus.api.plugins import PluginRegistry


class XRFMapsFileIO(FluorescenceFileReader, FluorescenceFileWriter):
    SIMPLE_NAME: Final[str] = 'XRF-Maps'

    def get_file_description(self) -> str:
        return 'XRF-Maps Fluorescence Dataset'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.h5*'}

    def read(self, file_path: Path) -> FluorescenceDataset:
        return load_fluorescence_data(file_path)

    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        save_fluorescence_data(file_path, dataset)


def register_plugins(registry: PluginRegistry) -> None:
    xrf_maps_file_io = XRFMapsFileIO()

    registry.fluorescence_file_readers.register_plugin(
        xrf_maps_file_io,
        simple_name=XRFMapsFileIO.SIMPLE_NAME,
    )
    registry.fluorescence_file_writers.register_plugin(
        xrf_maps_file_io,
        simple_name=XRFMapsFileIO.SIMPLE_NAME,
    )
