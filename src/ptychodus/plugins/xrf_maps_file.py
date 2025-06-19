from pathlib import Path
from typing import Final

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.typing import RealArrayType
from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceFileReader,
    FluorescenceFileWriter,
)


class XRFMapsFileIO(FluorescenceFileReader, FluorescenceFileWriter):
    SIMPLE_NAME: Final[str] = 'XRF-Maps'
    DISPLAY_NAME: Final[str] = 'XRF-Maps Fluorescence Dataset (*.h5 *.h5*)'

    @staticmethod
    def _split_path(data_path: str) -> tuple[str, str]:
        parts = data_path.split('/')
        return '/'.join(parts[:-1]), parts[-1]

    def read(self, file_path: Path) -> FluorescenceDataset:
        element_maps: list[ElementMap] = list()
        counts_per_second_path = str()
        channel_names_path = str()

        with h5py.File(file_path, 'r') as h5_file:
            # try to see if v10 layout, Non Negative Lease squares fitting tech was used
            h5_counts_per_second = h5_file['/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec']
            h5_channel_names = h5_file['/MAPS/XRF_Analyzed/NNLS/Channel_Names']

            if h5_counts_per_second is None:
                # try to see if v10 layout, iterative matrix fitting tech was used
                h5_counts_per_second = h5_file['/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec']
                h5_channel_names = h5_file['/MAPS/XRF_Analyzed/Fitted/Channel_Names']

                if h5_counts_per_second is None:
                    # try to see if was saved in v9 layout
                    h5_counts_per_second = h5_file['/MAPS/XRF_fits']
                    h5_channel_names = h5_file['/MAPS/channel_names']

            if h5_counts_per_second is not None:
                # Counts_Per_Sec is an N x H x W
                # where N is number of elements, use channel_names to find what element index
                counts_per_second = h5_counts_per_second[...]
                channel_names = h5_channel_names[...]

                for bname, cps in zip(channel_names, counts_per_second):
                    string_info = h5py.check_string_dtype(bname.dtype)
                    name = bname.decode(string_info.encoding)
                    emap = ElementMap(name, cps)
                    element_maps.append(emap)

            counts_per_second_path = h5_counts_per_second.name
            channel_names_path = h5_channel_names.name

        return FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=counts_per_second_path,
            channel_names_path=channel_names_path,
        )

    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        channel_names: list[str] = list()
        counts_per_sec: list[RealArrayType] = list()

        for emap in dataset.element_maps:
            channel_names.append(emap.name)
            counts_per_sec.append(emap.counts_per_second)

        cps_group_path, cps_dataset_name = self._split_path(dataset.counts_per_second_path)
        ch_group_path, ch_dataset_name = self._split_path(dataset.channel_names_path)

        with h5py.File(file_path, 'w') as h5_file:
            cps_group = h5_file.require_group(cps_group_path)
            cps_group.create_dataset(cps_dataset_name, data=numpy.stack(counts_per_sec))
            ch_group = h5_file.require_group(ch_group_path)
            ch_group.create_dataset(ch_dataset_name, data=channel_names, dtype='S256')


class NPZFluorescenceFileWriter(FluorescenceFileWriter):
    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        contents = {emap.name: emap.counts_per_second for emap in dataset.element_maps}
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
